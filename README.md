# ViralClip — готовый продукт (frontend + backend)

Клиенты **не вводят** API-ключ. Ключ лежит только на сервере в файле `.env`.

## Что внутри

```
viralclip-product/
  server.js          — Express: статика + /api/transcribe + /api/health
  package.json
  .env.example       — шаблон секретов
  .gitignore
  public/
    index.html       — сайт ViralClip
```

### Как работает анализ

1. **В браузере:** кадры, движение, склейки, громкость/тишина.
2. **На сервере:** речь через **OpenAI Whisper** (`POST /api/transcribe`).
3. Если сервер без ключа или упал — автоматический **fallback** на Whisper в браузере.

---

## Быстрый старт (на своём компьютере)

Нужен [Node.js 18+](https://nodejs.org/).

```bash
cd viralclip-product
cp .env.example .env
```

Открой `.env` и вставь ключ:

```env
OPENAI_API_KEY=sk-xxxxxxxx
PORT=3000
```

Ключ: https://platform.openai.com/api-keys

```bash
npm install
npm start
```

Открой в браузере: **http://localhost:3000**

---

## Деплой (чтобы сайт был в интернете)

Подойдут **Render**, **Railway**, **Fly.io**, **VPS**.

### Пример: Render

1. Залей папку `viralclip-product` на GitHub (файл `.env` **не** коммить — он в `.gitignore`).
2. New → Web Service → выбери репозиторий.
3. Build: `npm install`
4. Start: `npm start`
5. Environment → добавь `OPENAI_API_KEY` = твой `sk-...`
6. Deploy. Откроется URL вида `https://viralclip-xxxx.onrender.com`

### Пример: Railway

1. New Project → Deploy from GitHub.
2. Variables → `OPENAI_API_KEY`.
3. Start command: `npm start`.

### Свой VPS (Ubuntu)

```bash
cd viralclip-product
cp .env.example .env   # вписать ключ
npm install
npm start
# лучше через pm2:
# npm i -g pm2 && pm2 start server.js --name viralclip
```

Поставь nginx/caddy с HTTPS перед портом 3000.

---

## Важно про ключ

| Делать | Не делать |
|--------|-----------|
| Хранить в `.env` на сервере | Вставлять `sk-` в HTML |
| Добавлять в Environment Variables хостинга | Коммитить `.env` в git |
| При утечке — удалить ключ в OpenAI и создать новый | Скидывать ключ в чаты |

---

## API

### `GET /api/health`

```json
{ "ok": true, "service": "viralclip", "whisper": "openai" }
```

`whisper: "unavailable"` — ключ не задан.

### `POST /api/advise`

JSON: кадры (base64) + замеры → советы от GPT-4o-mini (vision).

### `POST /api/transcribe`

- `multipart/form-data`
- поле `file` — аудио/видео
- опционально `language` = `ru` | `en`

Ответ: `{ text, segments: [{start,end,text}], engine }`.

---

## Клиентам

Они просто открывают сайт и загружают видео. Никаких ключей, полей `sk-` и настроек OpenAI.


## Защита и телефон

- Лимиты по IP: 10 распознаваний/час, 10 AI-разборов/час, 30/день (настраивается в `.env`).
- Заголовки безопасности на сервере.
- Ключ OpenAI только в `.env`, не в HTML.
- **PWA:** на телефоне открой сайт → «На экран Домой» — ярлык как у приложения.
- Один бэкенд `/api/*` для сайта и будущего мобильного приложения.


## Payments (Stripe → your bank card)

Money does **not** go to a card number typed into the site. Flow:

1. User pays via **Stripe Checkout** (card form on Stripe).
2. Stripe sends money to **your Stripe balance**.
3. Stripe **payout** sends money to the **bank account** linked in Stripe (the account behind your Visa).

### Setup

1. Create account: https://dashboard.stripe.com/register  
2. Complete business / identity (may take 1–2 days).  
3. **Settings → Bank accounts and scheduling** — add the bank account for your Visa (or the account where salary lands).  
4. Developers → API keys → copy **Secret key** `sk_live_...` or test `sk_test_...`  
5. Put in `.env` or Render Environment:

```
STRIPE_SECRET_KEY=sk_test_...
PUBLIC_URL=https://your-app.onrender.com
ALLOW_DEV_PAYMENTS=0
```

6. Optional webhook: `https://your-app.onrender.com/api/billing/webhook`  
   Event: `checkout.session.completed`  
   Secret → `STRIPE_WEBHOOK_SECRET`

### Local test without Stripe

```
ALLOW_DEV_PAYMENTS=1
```

Checkout will **add coins without real charge** (for testing only).

### Uzbekistan note

If Stripe does not accept UZ as merchant country, use Payme/Click later or open Stripe via another supported country/entity. The code is ready for Stripe keys when you have them.


## Google Sign-In + Admin

1. Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID → **Web application**
2. Authorized JavaScript origins: `https://YOUR-app.onrender.com` and `http://localhost:3000`
3. Render Environment:
   - `GOOGLE_CLIENT_ID=....apps.googleusercontent.com`
   - `ADMIN_EMAILS=boburkachok2010@gmail.com`
4. Login as that Gmail → Admin panel at bottom of site
5. Note: on Render free tier `users.json` resets when the service redeploys/sleeps — for production use a real database later
