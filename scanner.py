import requests
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] 啟動台股全市場 (上市+上櫃) 雙核心掃描作業...")

# ---------------------------------------------------------
# 1. 動態取得全台股上市 (TWSE) 與 上櫃 (TPEx) 普通股清單
# ---------------------------------------------------------
def get_tw_all_tickers():
    tickers = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # 1. 上市普通股 (TWSE OpenAPI)
    try:
        twse_url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        r = requests.get(twse_url, headers=headers, timeout=10)
        if r.status_code == 200:
            for item in r.json():
                code = item.get("Code", "").strip()
                name = item.get("Name", "").strip()
                # 排除權證(長度>4)、ETF(00開頭)與特定憑證，只鎖定4位數普通股
                if len(code) == 4 and code.isdigit() and not code.startswith("00"):
                    tickers[f"{code}.TW"] = name
        print(f"  ✓ 上市股票取得完畢：共 {len([k for k in tickers if k.endswith('.TW')])} 檔")
    except Exception as e:
        print(f"  ✗ 上市股票清單抓取失敗: {e}")

    # 2. 上櫃普通股 (TPEx OpenAPI)
    try:
        tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
        r = requests.get(tpex_url, headers=headers, timeout=10)
        if r.status_code == 200:
            for item in r.json():
                code = item.get("SecuritiesCompanyCode", "").strip()
                name = item.get("CompanyName", "").strip()
                if len(code) == 4 and code.isdigit() and not code.startswith("00"):
                    tickers[f"{code}.TWO"] = name
        print(f"  ✓ 上櫃股票取得完畢：共 {len([k for k in tickers if k.endswith('.TWO')])} 檔")
    except Exception as e:
        print(f"  ✗ 上櫃股票清單抓取失敗: {e}")

    return tickers

stock_dict = get_tw_all_tickers()
all_tickers = list(stock_dict.keys())
total_count = len(all_tickers)
print(f"  ➜ 全市場普通股合計: {total_count} 檔")

if total_count == 0:
    print("無法取得股票清單，程序終止。")
    exit(1)

# ---------------------------------------------------------
# 2. 第一階：全市場向量化均線初篩 (Batch Download)
# ---------------------------------------------------------
print(f"[{datetime.now().strftime('%H:%M:%S')}] 執行第一階：全市場 20/60/200MA 多頭排列篩選...")

BATCH_SIZE = 100
stage1_passed = []

for i in range(0, total_count, BATCH_SIZE):
    batch = all_tickers[i:i + BATCH_SIZE]
    try:
        # 批量抓取日 K 線 (1年期)，大幅降低請求次數
        data = yf.download(batch, period="1y", interval="1d", group_by="ticker", threads=True, progress=False)
        
        for symbol in batch:
            try:
                # 兼容單一或多檔回傳結構
                df = data[symbol] if len(batch) > 1 else data
                df = df.dropna(subset=['Close'])
                if len(df) < 200:
                    continue

                close_series = df['Close']
                ma20 = close_series.rolling(20).mean()
                ma60 = close_series.rolling(60).mean()
                ma200 = close_series.rolling(200).mean()

                curr_close = close_series.iloc[-1]
                curr_20 = ma20.iloc[-1]
                curr_60 = ma60.iloc[-1]
                curr_200 = ma200.iloc[-1]
                prev_60 = ma60.iloc[-5] # 5天前季線

                # Minervini Stage 2 多頭排列：Close > 20MA > 60MA > 200MA 且 60MA 向上
                if (curr_close > curr_20 > curr_60 > curr_200) and (curr_60 >= prev_60):
                    stage1_passed.append((symbol, df))
            except Exception:
                continue
    except Exception as e:
        print(f"  批次下載異常 ({i}~{i+BATCH_SIZE}): {e}")
        
    time.sleep(1) # 溫和頻率控制

print(f"  ✓ 第一階均線多頭初篩完成：共 {len(stage1_passed)} 檔符合 Stage 2 結構")

# ---------------------------------------------------------
# 3. 第二階 (00981A 增長篩選) 與 第三階 (交易指導員 SMC 量化解析)
# ---------------------------------------------------------
print(f"[{datetime.now().strftime('%H:%M:%S')}] 執行第二階與第三階精篩 (00981A 財報 + Alpha Architect)...")

final_results = []

for symbol, df in stage1_passed:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # 00981A 財報指標
        roe = info.get("returnOnEquity", 0) or 0
        rev_growth = info.get("revenueGrowth", 0) or 0
        earn_growth = info.get("earningsGrowth", 0) or 0
        ocf = info.get("operatingCashflow", 0) or 0
        pe = info.get("trailingPE", 0) or info.get("forwardPE", 0) or 0
        
        # PEG 計算
        peg = info.get("pegRatio", None)
        if not peg and pe and earn_growth > 0:
            peg = pe / (earn_growth * 100)
        peg = peg if peg else 999

        # 00981A 篩選門檻：近四季 ROE >= 15%、營收或獲利成長 >= 10%、PEG < 1.5、現金流 OCF > 0
        growth_metric = max(rev_growth, earn_growth)
        if not ((roe >= 0.15) and (growth_metric >= 0.10) and (peg < 1.5) and (ocf > 0)):
            continue

        # 第三階：交易指導員-v20260611 量化規則計算
        close_price = float(df['Close'].iloc[-1])
        ma60_val = float(df['Close'].rolling(60).mean().iloc[-1])
        ma200_val = float(df['Close'].rolling(200).mean().iloc[-1])
        bias_60 = ((close_price - ma60_val) / ma60_val) * 100

        # SMC FVG 缺口檢測 (前3根K線)
        c1 = df.iloc[-3]
        c3 = df.iloc[-1]
        fvg_text = "未見顯著缺口"
        buy_zone = f"{close_price * 0.98:.1f} - {close_price:.1f}"
        if c3['Low'] > c1['High']:
            fvg_text = f"Bullish FVG: [{c1['High']:.1f} - {c3['Low']:.1f}]"
            buy_zone = f"{c1['High']:.1f} - {c3['Low']:.1f} (回補缺口介入)"

        # 60MA 季線乖離防線 (+15% 強制熔斷)
        bias_warning = False
        if bias_60 > 15.0:
            bias_warning = True
            buy_zone = "【強制關閉】60MA 季線乖離超過 +15%，嚴禁追高，等待回調"

        # 封裝近 250 日 K 線數據
        chart_data = []
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA200'] = df['Close'].rolling(200).mean()

        for idx, row in df.tail(250).iterrows():
            chart_data.append({
                "time": idx.strftime("%Y-%m-%d"),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "ma20": round(float(row['MA20']), 2) if not np.isnan(row['MA20']) else None,
                "ma60": round(float(row['MA60']), 2) if not np.isnan(row['MA60']) else None,
                "ma200": round(float(row['MA200']), 2) if not np.isnan(row['MA200']) else None
            })

        stock_name = stock_dict.get(symbol, info.get("shortName", symbol))
        final_results.append({
            "symbol": symbol,
            "name": stock_name,
            "close": close_price,
            "bias_60": round(bias_60, 2),
            "bias_warning": bias_warning,
            "fundamentals": {
                "roe": f"{roe * 100:.1f}%",
                "growth": f"{growth_metric * 100:.1f}%",
                "peg": f"{peg:.2f}",
                "ocf": f"{ocf / 1e8:.2f} 億"
            },
            "ict": {
                "fvg": fvg_text,
                "buy_zone": buy_zone,
                "stop_loss": f"{close_price * 0.95:.1f} (跌破波段低點)",
                "take_profit": f"{close_price * 1.15:.1f} (買方流動性磁鐵區)"
            },
            "chart_data": chart_data
        })
        print(f"  ★ 全市場雙篩精選入榜: {symbol} {stock_name} (成長率: {growth_metric*100:.1f}%, PEG: {peg:.2f})")

    except Exception as e:
        continue

# ---------------------------------------------------------
# 4. 產出給前端 GitHub Pages 讀取之 data.json
# ---------------------------------------------------------
output_payload = {
    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "market_scanned": total_count,
    "stage1_count": len(stage1_passed),
    "final_count": len(final_results),
    "stocks": final_results
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output_payload, f, ensure_ascii=False, indent=2)

print(f"[{datetime.now().strftime('%H:%M:%S')}] 全市場掃描任務完成！總掃描: {total_count} 檔，均線通過: {len(stage1_passed)} 檔，最終精選: {len(final_results)} 檔。已更新 data.json。")
