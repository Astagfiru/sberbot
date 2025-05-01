import os
import io
import json
import logging
import base64
from collections import Counter
from datetime import datetime, date, timedelta

import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
from langchain_community.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage
from dotenv import load_dotenv

from yandex_reviews_parser.parsers import Parser
from yandex_reviews_parser.utils import YandexParser
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")

# --- Патч для корректного получения рейтинга ---
_original_get = Parser._Parser__get_data_item


def _patched_get(self, elem):
    item = _original_get(self, elem)
    try:
        meta = elem.find_element(By.XPATH, ".//meta[@itemprop='ratingValue']")
        stars = float(meta.get_attribute("content"))
    except Exception:
        try:
            spans = elem.find_elements(By.CSS_SELECTOR, ".business-rating-badge-view__star._full")
            stars = len(spans)
        except Exception:
            stars = item.get("stars", 0)
    item["stars"] = stars
    return item


Parser._Parser__get_data_item = _patched_get

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Store parsed reviews in memory
reviews_cache = {}

# Initialize GigaChat
chat = GigaChat(credentials=GIGACHAT_AUTH_KEY, verify_ssl_certs=False)

# Load towns and banks data
with open("towns.json", "r", encoding="utf-8") as f:
    towns = json.load(f)

banks = []
with open("banks_with_addresses.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            bank_id, name, address = parts
            banks.append({
                "id": int(bank_id),
                "name": name,
                "address": address
            })


def clean_text(text: str) -> str:
    import re
    lines = []
    for line in text.splitlines():
        line = line.replace("**", "").replace("*", "")
        line = re.sub(r'^\s*#+\s*', "", line)
        lines.append(line)
    return "\n".join(lines).strip()


def analyze_reviews(reviews, city, branch_addr, start_dt, end_dt, period_desc):
    try:
        # Report parameters
        drange = f"{start_dt.strftime('%d.%m.%Y')} – {end_dt.strftime('%d.%m.%Y')}"
        report_settings = (
            f"Настройки отчёта:\n"
            f"🏙️ Город: {city}\n"
            f"🏦 Отделение: {branch_addr}\n"
            f"📆 Период: {drange}\n\n"
        )

        # Ensure reviews are properly formatted
        formatted_reviews = []
        for review in reviews:
            try:
                formatted_reviews.append({
                    'date': review.get('date', 0),
                    'stars': float(review.get('stars', 0)),
                    'text': review.get('text', ''),
                    'name': review.get('name', 'Аноним')
                })
            except (ValueError, TypeError):
                continue

        # Sort reviews by date
        formatted_reviews.sort(key=lambda x: x['date'])

        # Rating statistics
        cnt = Counter(int(r['stars']) for r in formatted_reviews)
        stars = sorted(cnt.items())
        stats_text = "\n".join(f"{star}⭐ — {count} шт." for star, count in stars)

        # Prepare timeline data
        timeline_data = []
        current_date = None
        current_sum = 0
        current_count = 0

        for review in formatted_reviews:
            review_date = datetime.fromtimestamp(review['date']).strftime('%d.%m.%Y')

            if current_date != review_date:
                if current_date is not None:
                    timeline_data.append({
                        'date': current_date,
                        'avg_rating': round(current_sum / current_count, 2) if current_count > 0 else 0
                    })
                current_date = review_date
                current_sum = review['stars']
                current_count = 1
            else:
                current_sum += review['stars']
                current_count += 1

        if current_date is not None:
            timeline_data.append({
                'date': current_date,
                'avg_rating': round(current_sum / current_count, 2) if current_count > 0 else 0
            })

        # Prepare rating distribution
        rating_dist = {i: 0 for i in range(1, 6)}
        for review in formatted_reviews:
            stars = int(review['stars'])
            if 1 <= stars <= 5:
                rating_dist[stars] += 1

        # Calculate percentages for pie chart
        total_reviews = len(formatted_reviews)
        rating_percentages = {
            star: round((count / total_reviews * 100), 1) if total_reviews > 0 else 0
            for star, count in rating_dist.items()
        }

        # Prepare chart data
        charts = {
            'timeline': {
                'labels': [item['date'] for item in timeline_data],
                'data': [item['avg_rating'] for item in timeline_data]
            },
            'bar': {
                'labels': [f"{i}⭐" for i in range(1, 6)],
                'data': [rating_dist[i] for i in range(1, 6)]
            },
            'pie': {
                'labels': [f"{i}⭐" for i in range(1, 6)],
                'data': [rating_percentages[i] for i in range(1, 6)]
            }
        }

        # Ensure we have at least one data point
        if not charts['timeline']['labels']:
            charts['timeline'] = {
                'labels': [start_dt.strftime('%d.%m.%Y')],
                'data': [0]
            }

        # GigaChat analysis
        filtered_reviews = "\n".join(
            f"{r['name']} ({datetime.fromtimestamp(r['date']).strftime('%d.%m.%Y')}): {r['text']} ({r['stars']}⭐)"
            for r in formatted_reviews
        )

        prompt = (
            f"Проанализируй отзывы клиентов банка за период {period_desc}:\n\n"
            f"{filtered_reviews}\n\n"
            "Сформируй отчёт строго в следующем формате, где X, Y и Z это проценты тональности отзывов:\n\n"
            "📅 Отчёт за {date}\n"
            "🔍 Всего отзывов: {count}\n"
            "⚠️ Топ-3 проблемы:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n\n"
            "📊 Настроение:\n"
            "😠 X% 😐 Y% 😊 Z%\n\n"
            "💡 Рекомендации:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ..."
        ).format(date=date.today().strftime("%d.%m.%Y"), count=len(formatted_reviews))

        response = chat.invoke([HumanMessage(content=prompt)])
        analysis = clean_text(response.content)

        return {
            'stats': stats_text,
            'charts': charts,
            'analysis': analysis
        }

    except Exception as e:
        logger.exception(f"Error in analyze_reviews: {str(e)}")
        return {
            'error': str(e),
            'stats': 'Ошибка при анализе статистики',
            'charts': {
                'timeline': {'labels': [start_dt.strftime('%d.%m.%Y')], 'data': [0]},
                'bar': {'labels': [f"{i}⭐" for i in range(1, 6)], 'data': [0] * 5},
                'pie': {'labels': [f"{i}⭐" for i in range(1, 6)], 'data': [0] * 5}
            },
            'analysis': 'Произошла ошибка при анализе отзывов'
        }


@app.route('/')
def index():
    return render_template('index.html', towns=towns)


@app.route('/get_branches/<city>')
def get_branches(city):
    branches = [b for b in banks if city.lower() in b["address"].lower()]
    return jsonify(branches)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        branch_id = data['branch_id']
        city = data['city']
        branch_addr = data['branch_address']
        period_type = data['period_type']
        custom_dates = data.get('custom_dates')

        # Get reviews from cache or parse if not available
        if branch_id not in reviews_cache:
            try:
                parser = YandexParser(branch_id)
                all_data = parser.parse(type_parse="reviews")
                reviews = all_data.get("company_reviews", [])
                reviews_cache[branch_id] = reviews
                logger.info(f"Successfully parsed {len(reviews)} reviews for branch {branch_id}")
            except Exception as e:
                logger.exception("Error parsing reviews")
                return jsonify({
                    'error': str(e),
                    'stats': 'Ошибка при получении отзывов',
                    'charts': {
                        'timeline': {'labels': [], 'data': []},
                        'bar': {'labels': [f"{i}⭐" for i in range(1, 6)], 'data': [0] * 5},
                        'pie': {'labels': [f"{i}⭐" for i in range(1, 6)], 'data': [0] * 5}
                    },
                    'analysis': 'Не удалось получить отзывы',
                    'reviews': []
                }), 500
        else:
            reviews = reviews_cache[branch_id]

        # Filter reviews by period
        now = datetime.now()

        if period_type == "last_24h":
            start_dt, end_dt = now - timedelta(days=1), now
            desc = "за последние 24 часа"
        elif period_type == "current_month":
            start_dt, end_dt = now.replace(day=1, hour=0, minute=0, second=0), now
            desc = "за текущий месяц"
        elif period_type == "last_30_days":
            start_dt, end_dt = now - timedelta(days=30), now
            desc = "за последние 30 дней"
        elif period_type == "last_3_months":
            start_dt, end_dt = now - timedelta(days=90), now
            desc = "за последние 3 месяца"
        elif period_type == "this_year":
            start_dt = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_dt = now
            desc = "за этот год"
        elif period_type == "custom" and custom_dates:
            start_str, end_str = custom_dates.split('-')
            start_dt = datetime.strptime(start_str.strip(), "%d.%m.%Y")
            end_dt = datetime.strptime(end_str.strip(), "%d.%m.%Y") + timedelta(hours=23, minutes=59, seconds=59)
            desc = f"за период {start_str}–{end_str}"
        elif period_type == "latest_10":
            reviews = sorted(reviews, key=lambda r: r.get("date", 0), reverse=True)[:10]
            start_dt = min(datetime.fromtimestamp(r["date"]) for r in reviews) if reviews else now
            end_dt = max(datetime.fromtimestamp(r["date"]) for r in reviews) if reviews else now
            desc = "10 новейших отзывов"
        else:
            return jsonify({'error': 'Invalid period type'}), 400

        if period_type != "latest_10":
            def in_range(r):
                return start_dt <= datetime.fromtimestamp(r.get("date", 0)) <= end_dt

            reviews = [r for r in reviews if in_range(r)]

        if not reviews:
            return jsonify({
                'stats': 'Нет отзывов за выбранный период',
                'charts': {
                    'timeline': {'labels': [start_dt.strftime('%d.%m.%Y')], 'data': [0]},
                    'bar': {'labels': [f"{i}⭐" for i in range(1, 6)], 'data': [0] * 5},
                    'pie': {'labels': [f"{i}⭐" for i in range(1, 6)], 'data': [0] * 5}
                },
                'analysis': 'Нет данных для анализа',
                'reviews': []
            })

        # Analyze reviews
        analysis_result = analyze_reviews(reviews, city, branch_addr, start_dt, end_dt, desc)

        # Return in the format expected by frontend
        return jsonify({
            'stats': analysis_result['stats'],
            'charts': analysis_result['charts'],
            'analysis': analysis_result['analysis'],
            'reviews': reviews
        })

    except Exception as e:
        logger.exception("Error in analyze endpoint")
        return jsonify({
            'error': str(e),
            'stats': 'Ошибка при анализе',
            'charts': {
                'timeline': {'labels': [], 'data': []},
                'bar': {'labels': [f"{i}⭐" for i in range(1, 6)], 'data': [0] * 5},
                'pie': {'labels': [f"{i}⭐" for i in range(1, 6)], 'data': [0] * 5}
            },
            'analysis': 'Произошла ошибка при обработке запроса',
            'reviews': []
        }), 500


if __name__ == '__main__':
    app.run(debug=True)