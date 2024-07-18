"""
Yandex CF Cron который стягивает ТА и отправляет в ТГ бота
"""
import asyncio
import datetime
import json
import os

from dotenv import load_dotenv
from telegram import Bot
from tradingview_ta import Interval

from tv_ta.config import BASE_DIR
from tv_ta.tickers import TICKERS
from tv_ta.api import TAAnalysisRepo, Analysis


async def handler(event, context):
    await main()
    return {
        "statusCode": 200,
        "body": "Hello World!",
    }


async def main(ignore_schedule=False):
    if not ignore_schedule and not _exchange_is_open():
        return

    load_dotenv(BASE_DIR / ".env")

    ta_repo = TAAnalysisRepo(Interval.INTERVAL_1_HOUR)

    ta: list[Analysis] = ta_repo.load_samples(TICKERS, dt=datetime.datetime.now(), sample=0)
    ta_json = [anal.model_dump(mode="json") for anal in ta]

    await _tg_bot_send_json(ta_json)


def _exchange_is_open(now_utc=None):
    now_utc = now_utc or datetime.datetime.utcnow()

    from_hour_utc = 10 - 3
    to_hour_utc = 19 - 3
    weekend = (5, 6)

    return from_hour_utc <= now_utc.hour <= to_hour_utc and now_utc.weekday() not in weekend


async def _tg_bot_send_json(json_list: list):
    TOKEN = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_BOT_CHAT_ID")

    json_bytes = json.dumps(json_list, indent=4).encode("utf-8")

    now_msk = datetime.datetime.utcnow() + datetime.timedelta(hours=3)
    now_msk_str = now_msk.strftime("%Y-%m-%d_%H-%M-%S")

    bot = Bot(token=TOKEN)
    res = await bot.send_document(
        chat_id=chat_id,
        document=json_bytes,
        filename=f"ta_{now_msk_str}.json",
    )

    print(res)


if __name__ == "__main__":
    asyncio.run(main(ignore_schedule=True))
