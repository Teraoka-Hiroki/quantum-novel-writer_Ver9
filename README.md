# ✦ Quantum Novel Assistant (Ver.9 React)

生成AI（Google Gemini）と数理最適化（Fixstars Amplify）を組み合わせた、人間共創型の小説執筆支援システムです。
ブラックボックス最適化（BBO）を用いることで、ユーザーの「好み」を学習し、最適な物語の構成要素を提案します。

## ✨ 特徴

- **ハイブリッド構成**: Python (Flask) のバックエンドと React (Vite) のモダンなフロントエンド。
- **AIによる発想支援**: Gemini APIを使用して、テーマに沿った多様なシーン候補を自動生成。
- **量子計算・最適化**: Fixstars Amplifyを使用し、ユーザーの評価に基づいたパラメータ最適化を実行。
- **ヒューマン・イン・ザ・ループ**: ユーザーが候補を評価（レーティング）し、AIがそれを学習して次回の提案を改善。

## 🛠 セットアップ

### ローカル実行
1. リポジトリをクローン
2. バックエンドの準備:
   ```bash
   pip install -r requirements.txt
   python backend/server.py