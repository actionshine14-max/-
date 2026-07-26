# OADP シナリオ分離型レースシミュレーター（試作版）

JRA出馬表テキストを入力し、OADPの「S1/S2/S3は着順違いではなく4角到達構造の違い」という原則に沿って、異なる入力世界を作り、同一のモンテカルロエンジンで走らせます。

## 起動
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 実装済み
- JRAプレーンテキストの基本パース
- FACTから基礎DERIVED/ESTIMATEを生成
- ハナ取得確率、先手参加率
- 前列圧発生/被害、序盤位置不発
- 差し接続、残存エネルギー、進路実現、失速率
- S1/S2/S3別モンテカルロ
- 4角順位平均/SD、勝率、複勝率
- シナリオ独立監査
- CSV/JSON出力

## 制限
OADPで係数が完全固定されていない部分は暫定式です。人気・オッズは物理計算とシナリオ発生確率に使用しません。
