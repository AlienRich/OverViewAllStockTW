import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime

# 監控觀察池（可自定義擴充為台股中型100、台灣50或全市場關注代碼）
TICKERS = [
    "2330.TW", "2454.TW", "2382.TW", "2308.TW", "3231.TW", 
    "6669.TW", "3661.TW", "3443.TW", "2376.TW", "2345.TW",
    "3034.TW", "2379.TW", "3037.TW", "2603.TW", "2609.TW"
]

results = []

print(f"[{datetime.now().strftime('%H:%M:%S')}] 開始執行全市場三階量化掃描...")

for ticker in TICKERS:
    try:
        # 抓取近 1 年歷史日 K 線
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 200:
            continue
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 計算均線
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA200'] = df['Close'].rolling(200).mean()

        latest = df.iloc[-1]
        prev_ma60 = df['MA60'].iloc[-5] # 5天前季線

        # ----------------------------------------------------
        # 第一階：均線系統初篩 (Stage 2 多頭排列)
        # 條件：現價 > 20MA > 60MA > 200MA 且 60MA 趨勢向上
        # ----------------------------------------------------
        ma_pass = (latest['Close'] > latest['MA20'] > latest['MA60'] > latest['MA200']) and (latest['MA60'] >= prev_ma60)
        if not ma_pass:
            continue

        # ----------------------------------------------------
        # 第二階：00981A 增長型基金經理人精篩 (財報成長與品質)
        # ----------------------------------------------------
        stock = yf.Ticker(ticker)
        info = stock.info
        
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

        # 00981A 條件檢核：ROE > 15%, 成長 > 10%, PEG < 1.5, OCF > 0
        growth_metric = max(rev_growth, earn_growth)
        growth_pass = (roe >= 0.15) and (growth_metric >= 0.10) and (peg < 1.5) and (ocf > 0)
        
        if not growth_pass:
            continue

        # ----------------------------------------------------
        # 第三階：交易指導員-v20260611 量化規則解析
        # ----------------------------------------------------
        close_price = float(latest['Close'])
        ma60_val = float(latest['MA60'])
        ma200_val = float(latest['MA200'])
        bias_60 = ((close_price - ma60_val) / ma60_val) * 100

        # SMC FVG 缺口檢測 (檢視最近3根K線)
        c1 = df.iloc[-3]
        c3 = df.iloc[-1]
        fvg_text = "未見明顯回補缺口"
        buy_zone = f"{close_price * 0.98:.1f} - {close_price:.1f}"
        
        if c3['Low'] > c1['High']:
            fvg_text = f"Bullish FVG: [{c1['High']:.1f} - {c3['Low']:.1f}]"
            buy_zone = f"{c1['High']:.1f} - {c3['Low']:.1f} (回補缺口介入)"

        # 60MA 乖離防線 (+15% 熔斷)
        bias_warning = False
        if bias_60 > 15.0:
            bias_warning = True
            buy_zone = "【強制關閉】60MA 季線乖離超過 +15%，嚴禁追高，等待回調"

        # 封裝 K 線圖資料 (近 250 個交易日)
        chart_data = []
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

        results.append({
            "symbol": ticker,
            "name": info.get("shortName", ticker),
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
                "stop_loss": f"{close_price * 0.95:.1f} (波段低點)",
                "take_profit": f"{close_price * 1.15:.1f} (流動性磁鐵區)"
            },
            "chart_data": chart_data
        })
        print(f"  ✓ {ticker} 通過全市場均線與 00981A 精篩")

    except Exception as e:
        print(f"  ✗ {ticker} 分析錯誤: {e}")

# 輸出 JSON
output_payload = {
    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "stocks": results
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output_payload, f, ensure_ascii=False, indent=2)

print(f"掃描完成，共有 {len(results)} 檔個股入選，已寫入 data.json。")
