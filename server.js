/**
 * ViralClip backend
 * - Static site + PWA
 * - POST /api/transcribe → Whisper
 * - POST /api/advise    → GPT vision tips
 * - Rate limits by IP
 */

require('dotenv').config();
const path = require('path');
const fs = require('fs');
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const OpenAI = require('openai');
const crypto = require('crypto');

const PORT = Number(process.env.PORT) || 3000;
// strip quotes/spaces — common .env mistake
const OPENAI_API_KEY = String(process.env.OPENAI_API_KEY || '')
  .trim()
  .replace(/^["']|["']$/g, '');
const CORS_ORIGIN = process.env.CORS_ORIGIN || true;

const LIMIT_TRANSCRIBE_PER_HOUR = Number(process.env.LIMIT_TRANSCRIBE_PER_HOUR || 20);
const LIMIT_ANALYZE_PER_HOUR = Number(process.env.LIMIT_ANALYZE_PER_HOUR || 20);
const LIMIT_ANALYZE_PER_DAY = Number(process.env.LIMIT_ANALYZE_PER_DAY || 60);

const COINS_PER_ANALYSIS = 10;
const SIGNUP_BONUS_COINS = 10;

const app = express();

app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  next();
});

app.use(cors({ origin: CORS_ORIGIN }));
app.use(express.json({ limit: '30mb' }));

const buckets = new Map();

function clientIp(req) {
  const xf = req.headers['x-forwarded-for'];
  if (typeof xf === 'string' && xf.length) return xf.split(',')[0].trim();
  return req.socket.remoteAddress || 'unknown';
}

function takeToken(ip, key, limit, windowMs) {
  const now = Date.now();
  const id = ip + '|' + key;
  let b = buckets.get(id);
  if (!b || now - b.start >= windowMs) {
    b = { start: now, count: 0 };
    buckets.set(id, b);
  }
  b.count += 1;
  const left = Math.max(0, limit - b.count);
  const ok = b.count <= limit;
  return {
    ok,
    left,
    limit,
    retryAfterSec: ok ? 0 : Math.ceil((b.start + windowMs - now) / 1000)
  };
}

function rateLimit(key, limit, windowMs) {
  return (req, res, next) => {
    const r = takeToken(clientIp(req), key, limit, windowMs);
    res.setHeader('X-RateLimit-Limit', String(r.limit));
    res.setHeader('X-RateLimit-Remaining', String(r.left));
    if (!r.ok) {
      return res.status(429).json({
        error: 'Too many requests. Try again later.',
        error_ru: 'Слишком много запросов. Попробуйте позже.',
        retryAfterSec: r.retryAfterSec,
        limit: r.limit
      });
    }
    next();
  };
}

const publicDir = path.join(__dirname, 'public');
app.use(
  express.static(publicDir, {
    setHeaders(res, filePath) {
      if (filePath.endsWith('sw.js')) {
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Service-Worker-Allowed', '/');
      }
      if (filePath.endsWith('.html')) {
        res.setHeader('Cache-Control', 'no-cache');
      }
    }
  })
);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const ok =
      /audio\//.test(file.mimetype) ||
      /video\//.test(file.mimetype) ||
      file.mimetype === 'application/octet-stream';
    cb(ok ? null : new Error('Only audio/video uploads allowed'), ok);
  }
});

function hasKey() {
  return OPENAI_API_KEY.startsWith('sk-') && OPENAI_API_KEY.length > 20;
}

function client() {
  return new OpenAI({ apiKey: OPENAI_API_KEY });
}

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'viralclip',
    whisper: hasKey() ? 'openai' : 'unavailable',
    advice: hasKey() ? 'openai-vision' : 'unavailable',
    limits: {
      transcribePerHour: LIMIT_TRANSCRIBE_PER_HOUR,
      analyzePerHour: LIMIT_ANALYZE_PER_HOUR,
      analyzePerDay: LIMIT_ANALYZE_PER_DAY
    }
  });
});

app.post(
  '/api/transcribe',
  rateLimit('transcribe_h', LIMIT_TRANSCRIBE_PER_HOUR, 60 * 60 * 1000),
  upload.single('file'),
  async (req, res) => {
    try {
      if (!hasKey()) {
        return res.status(503).json({
          error: 'OPENAI_API_KEY missing. Put sk-... into .env and restart npm start'
        });
      }
      if (!req.file?.buffer?.length) {
        return res.status(400).json({ error: 'No file uploaded (field: file)' });
      }

      const language = (req.body && req.body.language) || 'ru';
      const openai = client();
      const ext = guessExt(req.file.mimetype, req.file.originalname);
      const filename = `clip.${ext}`;

      console.log(
        '[transcribe]',
        clientIp(req),
        'bytes=',
        req.file.buffer.length,
        'lang=',
        language
      );

      let result;
      try {
        const file = await OpenAI.toFile(req.file.buffer, filename);
        result = await openai.audio.transcriptions.create({
          file,
          model: 'whisper-1',
          response_format: 'verbose_json',
          language: language === 'auto' ? undefined : language
        });
      } catch (e1) {
        console.warn('[transcribe] retry:', e1.message);
        const file2 = await OpenAI.toFile(req.file.buffer, filename);
        result = await openai.audio.transcriptions.create({
          file: file2,
          model: 'whisper-1',
          response_format: 'verbose_json'
        });
      }

      const segments = (result.segments || [])
        .map((s) => ({
          start: s.start,
          end: s.end,
          text: (s.text || '').trim()
        }))
        .filter((s) => s.text);

      res.json({
        text: (result.text || '').trim(),
        segments,
        language: result.language || language || null,
        engine: 'openai-whisper-1'
      });
    } catch (err) {
      const msg = err?.message || String(err);
      console.error('[transcribe] error:', msg);
      res.status(err?.status || 500).json({ error: msg });
    }
  }
);

app.post(
  '/api/advise',
  rateLimit('analyze_h', LIMIT_ANALYZE_PER_HOUR, 60 * 60 * 1000),
  rateLimit('analyze_d', LIMIT_ANALYZE_PER_DAY, 24 * 60 * 60 * 1000),
  async (req, res) => {
    try {
      if (!hasKey()) {
        return res.status(503).json({
          error: 'OPENAI_API_KEY missing. Put sk-... into .env and restart npm start'
        });
      }

      const body = req.body || {};
      const language = body.language === 'en' ? 'en' : (body.language === 'uz' ? 'uz' : 'ru');
      const duration = Number(body.duration) || 0;
      const frames = Array.isArray(body.frames) ? body.frames.slice(0, 8) : [];
      const metrics = body.metrics || {};
      const motion = body.motion || {};
      const audio = body.audio || {};
      const transcript = body.transcript || {};

      const paidUser = authUser(req);
      const devPay = String(process.env.ALLOW_DEV_PAYMENTS || '') === '1';
      if (!paidUser && !devPay) {
        return res.status(401).json({
          error: 'Login required',
          error_ru: 'Войдите в аккаунт, чтобы запускать анализ',
          code: 'AUTH'
        });
      }
      const bal = paidUser ? (Number(paidUser.coins) || 0) : 999;
      if (paidUser && bal < COINS_PER_ANALYSIS) {
        return res.status(402).json({
          error: 'Not enough coins',
          error_ru: 'Нужно ' + COINS_PER_ANALYSIS + ' монет. На балансе: ' + bal,
          code: 'COINS',
          coins: bal,
          need: COINS_PER_ANALYSIS
        });
      }

      if (!frames.length) {
        return res.status(400).json({ error: 'frames[] required' });
      }

      console.log('[analyze]', clientIp(req), 'frames=', frames.length, 'dur=', duration);

      const metricsBlock = [
        `Duration: ${duration.toFixed(1)}s`,
        `Scores: hook=${metrics.hook}, pacing=${metrics.pacing}, captions=${metrics.captions}, audio=${metrics.audio}, cta=${metrics.cta}, thumb=${metrics.thumb}`,
        `Motion: mean=${Number(motion.meanMotion || 0).toFixed(3)}, hook=${Number(motion.hookMotion || 0).toFixed(3)}, cuts/s=${Number(motion.cutsPerSec || 0).toFixed(2)}, cuts=${JSON.stringify((motion.cuts || []).slice(0, 15))}`,
        `Audio: silence=${Number(audio.silenceRatio || 0).toFixed(2)}, meanRms=${Number(audio.meanRms || 0).toFixed(3)}, hookRms=${Number(audio.hookRms || 0).toFixed(3)}`,
        `Transcript: ${(transcript.text || '').slice(0, 900)}`,
        `Segments: ${JSON.stringify((transcript.segments || []).slice(0, 14))}`
      ].join('\n');

      const systemRu = `MASTER: PROFESSIONAL VIDEO EDITING ANALYSIS ENGINE (ViralClip)

РОЛЬ
Ты — профессиональный видеомонтажёр и режиссёр монтажа: short-form (Reels/Shorts/TikTok), talking-head, education, memes, cinematic, commercial, podcast/interview, documentary, gaming, music video, motivational, lifestyle/vlog, product, events/sports, storytelling.
Мыслишь как монтажёр, отвечающий за результат. Каждое решение должно иметь ФУНКЦИЮ (ритм, смысл, эмоция, непрерывность, композиция, структура). Если элемент не полезен — не рекомендуй.
Не делай каждое видео «максимально вирусным TikTok». Стиль = жанр + аудитория + задача ролика.

ИЕРАРХИЯ ПРИОРИТЕТОВ
1) Смысл и достоверность  2) Понятность  3) Структура  4) Эмоция  5) Ритм/удержание  6) Визуальная непрерывность  7) Качество картинки/звука  8) Художественный стиль.
Не жертвуй смыслом ради динамики, понятностью ради эффектов.

СНАЧАЛА ЖАНР
Определи: основной/вторичный жанр, цель (учить/развлекать/продавать/история), тон, вертикаль/горизонталь, темп речи. Гибриды учитывай. Не угадывай платформу/статистику, если нет данных.

АНАЛИЗ ИСХОДНИКА
Опирайся на кадры, метрики, транскрипт, таймкоды. Не выдумывай содержимое, ошибки и таймкоды.
Разделяй: подтверждённое / вероятное / творческое предложение / нельзя проверить.
Если ролик уже хорош — скажи. Не ищи 20 проблем ради объёма.

УДЕРЖАНИЕ (не = вспышки каждые 0.5с)
• Хук: причина остаться в первые секунды (факт/вопрос/проблема/действие/результат) — по жанру.
• Не режь вступление автоматически, если оно нужно для смысла.
• Обещание начала должно выполняться; без ложных тизеров.
• Развитие: каждый блок добавляет мысль/эмоцию/доказательство.
• Смена кадра — когда есть новая информация/эмоция/действие, не «потому что прошло N секунд».
• Финал по жанру: вывод / результат / шутка / CTA только если уместен. Без пустого хвоста.

РИТМ И ПАУЗЫ
Различай скорость склеек и ощущение динамики.
Паузы: технические/ошибочные (обычно режь) vs для понимания vs эмоциональные (сохраняй).
Не делай речь неестественно плотной. Длительность кадра = содержание, не единый шаблон.
Контраст темпа полезен; разрыв шаблона — только если усиливает интерес.

СКЛЕЙКИ
Jump cut ок для темпа/юмора; следи за подёргиванием головы — иначе другая точка реза / B-roll / punch-in / J-L cut.
Match cut, cut on action, 180° — когда помогают пространственной логике.
J-cut / L-cut — для интервью, док, переходов. Прямая склейка часто лучше сложного перехода.
Не transition на каждую склейку.

КРУПНОСТЬ / КАДР
Планы по задаче (общий/средний/крупный/деталь). Punch-in — на тезис/реакцию, не на каждое предложение. Следи за safe-zone вертикали, текст vs лицо, цифровой зум и качество.

B-ROLL
Только смысловая связь с речью. Не закрывай лицо без нужды. Для сложного — схемы/шаги/подсветка, не хаос.

СУБТИТРЫ
Точность, синхрон, короткие смысловые группы, контраст, не на лице/UI. Анимация слов — точечно. Не выделяй всё. Разделяй субтитры / заголовки / CTA.

ЗВУК
Приоритет речи. Музыка не конкурирует с VO. SFX — функция (движение/юмор/переход), не на каждый cut. Тишина может быть сильнее громкого удара. Без выдуманных замечаний, если звук не дан в данных.

ЖАНРОВЫЕ АКЦЕНТЫ (кратко)
• Expert/talking-head: ясность тезиса, доверие, умеренные зумы, B-roll по смыслу.
• Education: шаги не вырезать критичные; UI читаем.
• Meme/comedy: setup → punchline → reaction; тайминг важнее громкости; не перегружай shake+emoji+zoom сразу.
• Commercial: проблема → решение → proof → CTA читаемый.
• Podcast/interview: контекст цитаты, не вырывать смысл.
• Cinematic/doc: эмоция и достоверность важнее «динамики ради динамики».

ТИПИЧНЫЕ ОШИБКИ (отмечай только если видно)
Слабый старт, повторы, мёртвые паузы, обрыв слов, субтитры на лице, музыка громче речи, бессмысленные transitions/эффекты, нет финала/CTA когда нужен.

ПРИОРИТЕТЫ ПРАВОК
high = P0/P1 (мешает смотреть/понимать или сильный провал хука/структуры/звука/субтитров)
mid = P2 (заметное улучшение)
low = P3 (полировка)

ЗАПРЕТЫ
• Нет воды: «improve engagement», «make more dynamic» без КОНКРЕТИКИ.
• Нет гарантий просмотров/виральности.
• Не путай вкус с ошибкой.
• Минимально достаточное исправление лучше сложного композитинга.
• 6–8 точных советов лучше 20 общих.
• Сохраняй то, что уже работает.

ФОРМАТ ОТВЕТА — СТРОГО JSON (без markdown):
{
  "overall_comment": "4–7 предложений RU: жанр/задача; что уже хорошо; главные 1–2 проблемы; главный вердикт",
  "genre": "кратко жанр",
  "tips": [
    {
      "t": 1.2,
      "title": "короткий заголовок",
      "advice": "2–4 предложения: (1) что на кадре/в речи (2) в чём проблема (3) что сделать. БЕЗ финальной фразы «Это сделает/поможет...».",
      "severity": "hook|pacing|captions|audio|cta|thumb|visual|music|structure|color",
      "priority": "high|mid|low"
    }
  ]
}
Ровно 6–8 tips. t из переданных кадров или ближайшее. Язык: русский.
Запрети копипаст: каждый tip про РАЗНУЮ проблему. Не пиши один и тот же текст про «мало движения» на несколько таймкодов.
Если движение реально слабое — один tip на самый слабый участок + конкретный приём под ЖАНР и речь из transcript.
КРИТИЧНО: advice не короче ~3 предложений. Одна короткая строка = плохой ответ. Сначала наблюдение по кадру, потом правка.

ВРЕМЕННАЯ СВЯЗНОСТЬ (очень важно):
• Смотри ВСЕ переданные кадры и транскрипт как одну последовательность, не изолированные скриншоты.
• Если на кадре t проблема (нет подписи «кто это», нет контекста, нет CTA) — проверь кадры и речь на t+1…t+4 с.
• Если ответ/анимация/пояснение УЖЕ идут сразу после — НЕ рекомендуй добавлять то, что уже есть. Напиши, что момент закрыт следующим кадром, или выбери другую реальную проблему.
• Советуй только то, чего в ролике реально нет на всём отрезке, а не «дыру» длиной 0.5–2 с перед уже готовым решением.
• Пример ошибки: имя «Набиев Сардор» без роли на кадре 12.0, но на 12.5–14.0 уже анимация достижений — совет «добавь кто он» ЗАПРЕЩЁН.`;

      const systemEn = `MASTER: PROFESSIONAL VIDEO EDITING ANALYSIS ENGINE (ViralClip)

You are a professional editor/director of montage across short-form, talking-head, education, meme, cinematic, commercial, podcast, documentary, gaming, music video, motivational, lifestyle, product, events.

Every recommendation must have a FUNCTION. Do not force viral TikTok style on every clip. Match genre + goal.

Priority order: meaning/truth → clarity → structure → emotion → rhythm/retention → continuity → technical AV → style.

First identify genre/goal/tone. Use frames + metrics + transcript. Never invent content, faults, or timestamps. Separate confirmed vs probable vs creative vs unverifiable. Praise what already works.

Retention ≠ constant flashes. Strong open (genre-appropriate), keep promise, progressive development, cuts when new info/emotion, fitting ending (CTA only if appropriate).

Rhythm: distinguish cut rate from perceived pace; keep comprehension/emotion pauses; don't over-tighten speech.

Cuts: meaningful jump cuts; watch head pops; prefer clean cuts over gimmick transitions; J/L-cuts when they help interviews/story.

Framing: shot size by purpose; punch-in on real beats not every sentence; vertical safe zones; captions not on face/UI.

B-roll only if it illustrates the line. Captions: accurate, short chunks, contrast, sparse word emphasis.

Audio: voice first; music ducked; SFX with function; silence can be stronger than a hit. Don't invent audio notes without evidence.

Genre notes: expert=clarity/trust; education=keep critical steps; comedy=timing over noise; commercial=clear offer+readable CTA; interview=preserve context.

Priorities: high=P0/P1, mid=P2, low=P3. Ban vague advice and view guarantees. 6–8 specific tips beat 20 generic ones.

JSON only:
{"overall_comment":"4-7 sentences: genre, what works, main issues, verdict","genre":"short","tips":[{"t":1.2,"title":"...","advice":"observation → problem → concrete edit → why","severity":"hook|pacing|captions|audio|cta|thumb|visual|music|structure|color","priority":"high|mid|low"}]}
6–8 tips. English.
CRITICAL: each advice at least ~3 sentences. Observation first, then concrete fix. One-liners are invalid.

TEMPORAL CONTINUITY:
• Treat all frames + transcript as one timeline, not isolated stills.
• If something seems missing at t, check t+1…t+4s. If the video already answers (animation, title, explanation), do NOT recommend adding it.
• Only flag gaps that stay unresolved across the segment, not 0.5–2s beats before an existing payoff.`;

      const hardRu = `ЖЁСТКИЕ ТРЕБОВАНИЯ К ОТВЕТУ:
1) overall_comment: 4–7 предложений — жанр, что УЖЕ хорошо, 1–2 главные проблемы, вердикт. Без воды.
2) genre: коротко (например «expert talking-head / education»).
3) Ровно 6–8 tips. Каждый advice ОБЯЗАТЕЛЬНО 3–5 предложений в структуре:
   НАБЛЮДЕНИЕ: что видно на кадре / слышно в речи в эту секунду (конкретика: лицо, текст на экране, жест, пауза).
   ПРОБЛЕМА: почему это мешает (понятность / ритм / удержание / жанр).
   ПРАВКА: точное действие в монтаже (вырезать X с–Y с, punch-in, перенести текст, B-roll какой именно, SFX где).
   ЗАЧЕМ: какой эффект для зрителя.
4) ЗАПРЕЩЕНО: однострочники без наблюдения; ЗАПРЕЩЕНО заканчивать каждый совет шаблоном «Это сделает/поможет/повысит...».
5) Если на кадре всё ок по теме — не выдумывай проблему; возьми другой слабый момент.
6) t привяжи к переданным кадрам.
7) ПЕРЕД советом «добавь пояснение / кто это / контекст» — проверь следующие кадры и транскрипт. Если пояснение уже есть через 1–3 с — этот совет ЗАПРЕЩЁН.
Только JSON.`;

      const hardEn = `HARD REQUIREMENTS:
1) overall_comment: 4–7 sentences — genre, what already works, 1–2 main issues, verdict.
2) genre: short label.
3) Exactly 6–8 tips. Each advice MUST be 3–5 sentences:
   OBSERVATION: what is visible/heard at that second.
   PROBLEM: why it hurts clarity/rhythm/retention.
   FIX: concrete edit action.
   WHY: viewer effect.
4) Ban one-liners like "improve the hook" without observation.
5) Do not invent problems; pick real weak moments.
6) t must match provided frames.
7) Before advising "add who this is / context", check the next 1–3s of frames/transcript. If already explained — ban that tip.
JSON only.`;

      const content = [
        {
          type: 'text',
          text:
            (language === 'en' ? hardEn : hardRu) +
            (language === 'uz' ? '\n\nJAVOB TILINI: o\'zbek tilida yozing (JSON maydonlari o\'zbekcha).\n' : '') +
            '\n\n' +
            metricsBlock +
            '\n\nFrames (look at each image):'
        }
      ];

      frames.forEach((f, i) => {
        const t = typeof f.t === 'number' ? f.t : i;
        let url = f.dataUrl || f.url || '';
        if (!url) return;
        if (!String(url).startsWith('data:')) url = 'data:image/jpeg;base64,' + url;
        content.push({ type: 'text', text: `\n#${i + 1} t=${Number(t).toFixed(2)}s` });
        content.push({ type: 'image_url', image_url: { url, detail: 'low' } });
      });

      const openai = client();
      const completion = await openai.chat.completions.create({
        model: process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini',
        temperature: 0.2,
        max_tokens: 3200,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: language === 'en' ? systemEn : systemRu },
          { role: 'user', content }
        ]
      });

      const raw = completion.choices?.[0]?.message?.content || '{}';
      let parsed;
      try {
        parsed = JSON.parse(raw);
      } catch {
        parsed = { overall_comment: raw, tips: [] };
      }

      const tips = Array.isArray(parsed.tips) ? parsed.tips : [];
      const order = { high: 0, mid: 1, low: 2 };
      const sorted = tips.slice().sort((a, b) => (order[a.priority] ?? 1) - (order[b.priority] ?? 1));
      let coinsLeft = null;
      let charged = 0;
      if (paidUser) {
        const after = updateUser(paidUser.email, (u) => {
          u.coins = Math.max(0, (Number(u.coins) || 0) - COINS_PER_ANALYSIS);
          u.analysesUsed = (u.analysesUsed || 0) + 1;
        });
        coinsLeft = after ? after.coins : bal - COINS_PER_ANALYSIS;
        charged = COINS_PER_ANALYSIS;
      }
      res.json({
        overall_comment: parsed.overall_comment || '',
        genre: parsed.genre || '',
        tips: sorted.map((tip) => ({
          t: Number(tip.t) || 0,
          title: String(tip.title || '').slice(0, 140),
          advice: String(tip.advice || '').slice(0, 1600),
          text: [tip.title, tip.advice].filter(Boolean).join('. ').slice(0, 1700),
          key: String(tip.severity || tip.key || 'pacing').slice(0, 32),
          severity: String(tip.severity || 'pacing').slice(0, 32),
          priority: String(tip.priority || 'mid').slice(0, 8)
        })),
        engine: process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini',
        coinsLeft,
        charged,
        guest: !paidUser
      });
    } catch (err) {
      const msg = err?.message || String(err);
      console.error('[analyze] error:', msg);
      res.status(err?.status || 500).json({ error: msg });
    }
  }
);

function guessExt(mime, name) {
  if (name && /\.[a-z0-9]+$/i.test(name)) {
    return name.split('.').pop().toLowerCase().replace(/[^a-z0-9]/g, '') || 'webm';
  }
  if (mime === 'audio/wav' || mime === 'audio/x-wav') return 'wav';
  if (mime === 'audio/mpeg' || mime === 'audio/mp3') return 'mp3';
  if (mime === 'audio/mp4' || mime === 'video/mp4') return 'mp4';
  if (mime === 'audio/webm' || mime === 'video/webm') return 'webm';
  if (mime === 'audio/m4a' || mime === 'audio/x-m4a') return 'm4a';
  return 'webm';
}

// ========== Data dir (waitlist + accounts) ==========
const dataDir = path.join(__dirname, 'data');
if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

function readJson(file, fallback) {
  try {
    const p = path.join(dataDir, file);
    if (!fs.existsSync(p)) return fallback;
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch {
    return fallback;
  }
}
function writeJson(file, data) {
  fs.writeFileSync(path.join(dataDir, file), JSON.stringify(data, null, 2), 'utf8');
}

function hashPassword(password, salt) {
  salt = salt || crypto.randomBytes(16).toString('hex');
  const hash = crypto.scryptSync(String(password), salt, 32).toString('hex');
  return { salt, hash };
}
function verifyPassword(password, salt, hash) {
  try {
    const h = crypto.scryptSync(String(password), String(salt), 32).toString('hex');
    const a = Buffer.from(h, 'hex');
    const b = Buffer.from(String(hash), 'hex');
    if (a.length !== b.length) return false;
    return crypto.timingSafeEqual(a, b);
  } catch {
    return false;
  }
}

const TOKEN_SECRET = process.env.TOKEN_SECRET || OPENAI_API_KEY || 'viralclip-dev-secret';
function signToken(payload) {
  const body = Buffer.from(JSON.stringify({ ...payload, exp: Date.now() + 30 * 24 * 3600 * 1000 })).toString('base64url');
  const sig = crypto.createHmac('sha256', TOKEN_SECRET).update(body).digest('base64url');
  return body + '.' + sig;
}
function verifyToken(token) {
  if (!token || typeof token !== 'string' || !token.includes('.')) return null;
  const [body, sig] = token.split('.');
  const expect = crypto.createHmac('sha256', TOKEN_SECRET).update(body).digest('base64url');
  if (sig !== expect) return null;
  try {
    const data = JSON.parse(Buffer.from(body, 'base64url').toString('utf8'));
    if (!data.exp || data.exp < Date.now()) return null;
    return data;
  } catch {
    return null;
  }
}
function authUser(req) {
  const h = req.headers.authorization || '';
  const token = h.startsWith('Bearer ') ? h.slice(7) : (req.headers['x-auth-token'] || '');
  const data = verifyToken(token);
  if (!data || !data.email) return null;
  const users = readJson('users.json', []);
  return users.find((u) => u.email === data.email) || null;
}

// ========== Coins & billing ==========
const PRODUCTS = {
  pack50:  { kind: 'pack', coins: 50,  amountUsd: 6,  name: 'Starter pack 50 coins' },
  pack150: { kind: 'pack', coins: 150, amountUsd: 15, name: 'Creator pack 150 coins' },
  pack400: { kind: 'pack', coins: 400, amountUsd: 35, name: 'Studio pack 400 coins' },
  starter: { kind: 'sub',  coins: 100, amountUsd: 10, name: 'Starter monthly', interval: 'month' },
  pro:     { kind: 'sub',  coins: 200, amountUsd: 16, name: 'Pro monthly', interval: 'month' },
  studio:  { kind: 'sub',  coins: 350, amountUsd: 24, name: 'Studio monthly + Instagram account audit', interval: 'month' }
};

function publicUser(u) {
  if (!u) return null;
  return {
    id: u.id,
    email: u.email,
    name: u.name,
    coins: Number(u.coins) || 0,
    plan: u.plan || 'free',
    planExpires: u.planExpires || null,
    analysesUsed: u.analysesUsed || 0
  };
}

function updateUser(email, mutator) {
  const users = readJson('users.json', []);
  const i = users.findIndex((x) => x.email === email);
  if (i < 0) return null;
  mutator(users[i]);
  writeJson('users.json', users);
  return users[i];
}

function getStripe() {
  const key = process.env.STRIPE_SECRET_KEY || '';
  if (!key || key.includes('...')) return null;
  try {
    return require('stripe')(key);
  } catch (e) {
    console.warn('stripe load failed', e.message);
    return null;
  }
}



// Waitlist
app.post('/api/waitlist', (req, res) => {
  const email = String((req.body && req.body.email) || '')
    .trim()
    .toLowerCase();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).json({ error: 'Invalid email', error_ru: 'Некорректный email' });
  }
  const list = readJson('waitlist.json', []);
  if (!list.find((x) => x.email === email)) {
    list.push({ email, at: new Date().toISOString(), ip: clientIp(req) });
    writeJson('waitlist.json', list);
  }
  res.json({ ok: true, message: 'joined' });
});

// Register
app.post('/api/auth/register', (req, res) => {
  const email = String((req.body && req.body.email) || '')
    .trim()
    .toLowerCase();
  const password = String((req.body && req.body.password) || '');
  const name = String((req.body && req.body.name) || '').trim().slice(0, 60);
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).json({ error: 'Invalid email', error_ru: 'Некорректный email' });
  }
  if (password.length < 6) {
    return res.status(400).json({
      error: 'Password min 6 characters',
      error_ru: 'Пароль минимум 6 символов'
    });
  }
  const users = readJson('users.json', []);
  if (users.find((u) => u.email === email)) {
    return res.status(409).json({
      error: 'Email already registered',
      error_ru: 'Этот email уже зарегистрирован'
    });
  }
  const { salt, hash } = hashPassword(password);
  const user = {
    id: crypto.randomBytes(8).toString('hex'),
    email,
    name: name || email.split('@')[0],
    salt,
    hash,
    createdAt: new Date().toISOString(),
    analysesUsed: 0,
    coins: SIGNUP_BONUS_COINS,
    plan: 'free',
    planExpires: null
  };
  users.push(user);
  writeJson('users.json', users);
  // also put on waitlist
  const list = readJson('waitlist.json', []);
  if (!list.find((x) => x.email === email)) {
    list.push({ email, at: user.createdAt, source: 'register' });
    writeJson('waitlist.json', list);
  }
  const token = signToken({ email: user.email, id: user.id });
  res.json({
    ok: true,
    token,
    user: publicUser(user)
  });
});

// Login
app.post('/api/auth/login', (req, res) => {
  const email = String((req.body && req.body.email) || '')
    .trim()
    .toLowerCase();
  const password = String((req.body && req.body.password) || '');
  const users = readJson('users.json', []);
  const user = users.find((u) => u.email === email);
  if (!user || !verifyPassword(password, user.salt, user.hash)) {
    return res.status(401).json({
      error: 'Wrong email or password',
      error_ru: 'Неверный email или пароль'
    });
  }
  const token = signToken({ email: user.email, id: user.id });
  res.json({
    ok: true,
    token,
    user: { id: user.id, email: user.email, name: user.name }
  });
});

// Me
app.get('/api/auth/me', (req, res) => {
  const user = authUser(req);
  if (!user) return res.status(401).json({ error: 'Unauthorized' });
  res.json({ ok: true, user: publicUser(user), products: PRODUCTS, coinsPerAnalysis: COINS_PER_ANALYSIS });
});
// ----- Billing: catalog -----
app.get('/api/billing/catalog', (req, res) => {
  res.json({
    coinsPerAnalysis: COINS_PER_ANALYSIS,
    signupBonus: SIGNUP_BONUS_COINS,
    products: PRODUCTS,
    stripeEnabled: !!(process.env.STRIPE_SECRET_KEY && !String(process.env.STRIPE_SECRET_KEY).includes('...'))
  });
});

// ----- Checkout (Stripe) -----
app.post('/api/billing/checkout', async (req, res) => {
  try {
    const user = authUser(req);
    if (!user) {
      return res.status(401).json({
        error: 'Login required',
        error_ru: 'Сначала войдите в аккаунт'
      });
    }
    const productId = String((req.body && req.body.productId) || '');
    const product = PRODUCTS[productId];
    if (!product) {
      return res.status(400).json({ error: 'Unknown product', error_ru: 'Неизвестный тариф' });
    }

    const stripe = getStripe();
    const base = (process.env.PUBLIC_URL || '').replace(/\/$/, '') || `${req.protocol}://${req.get('host')}`;

    if (!stripe) {
      // Dev fallback: credit coins immediately (NO real charge) when Stripe not configured
      if (process.env.ALLOW_DEV_PAYMENTS === '1') {
        const updated = updateUser(user.email, (u) => {
          u.coins = (Number(u.coins) || 0) + product.coins;
          if (product.kind === 'sub') {
            u.plan = productId;
            u.planExpires = new Date(Date.now() + 30 * 24 * 3600 * 1000).toISOString();
          }
        });
        return res.json({
          ok: true,
          dev: true,
          user: publicUser(updated),
          message: 'DEV payment: coins added (Stripe not configured)'
        });
      }
      return res.status(503).json({
        error: 'Payments not configured',
        error_ru: 'Оплата ещё не настроена: добавьте STRIPE_SECRET_KEY на сервере',
        setup: 'https://dashboard.stripe.com/apikeys'
      });
    }

    const session = await stripe.checkout.sessions.create({
      mode: product.kind === 'sub' ? 'subscription' : 'payment',
      customer_email: user.email,
      line_items: [
        {
          price_data: {
            currency: 'usd',
            unit_amount: Math.round(product.amountUsd * 100),
            product_data: { name: product.name },
            ...(product.kind === 'sub'
              ? { recurring: { interval: 'month' } }
              : {})
          },
          quantity: 1
        }
      ],
      metadata: {
        userId: user.id,
        email: user.email,
        productId,
        coins: String(product.coins),
        kind: product.kind
      },
      success_url: `${base}/?paid=1&product=${productId}`,
      cancel_url: `${base}/?paid=0`
    });

    res.json({ ok: true, url: session.url, sessionId: session.id });
  } catch (err) {
    console.error('[checkout]', err.message);
    res.status(500).json({ error: err.message || 'Checkout failed' });
  }
});

// Stripe webhook — add coins after successful payment
app.post('/api/billing/webhook', express.raw({ type: 'application/json' }), (req, res) => {
  const stripe = getStripe();
  if (!stripe) return res.status(503).send('stripe off');

  let event = req.body;
  const sig = req.headers['stripe-signature'];
  const whsec = process.env.STRIPE_WEBHOOK_SECRET || '';
  try {
    if (whsec && sig) {
      event = stripe.webhooks.constructEvent(req.body, sig, whsec);
    } else if (Buffer.isBuffer(req.body)) {
      event = JSON.parse(req.body.toString('utf8'));
    }
  } catch (err) {
    console.error('[webhook] signature', err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  try {
    if (event.type === 'checkout.session.completed') {
      const session = event.data.object;
      const email = (session.metadata && session.metadata.email) || session.customer_email;
      const productId = session.metadata && session.metadata.productId;
      const coins = Number((session.metadata && session.metadata.coins) || 0);
      const kind = (session.metadata && session.metadata.kind) || 'pack';
      if (email && coins > 0) {
        updateUser(String(email).toLowerCase(), (u) => {
          u.coins = (Number(u.coins) || 0) + coins;
          if (kind === 'sub' && productId) {
            u.plan = productId;
            u.planExpires = new Date(Date.now() + 32 * 24 * 3600 * 1000).toISOString();
          }
        });
        console.log('[webhook] credited', email, coins, productId);
      }
    }
  } catch (e) {
    console.error('[webhook] handler', e);
  }
  res.json({ received: true });
});



// ----- Instagram / account audit (Studio premium) -----
app.post('/api/account-audit', async (req, res) => {
  try {
    if (!hasKey()) {
      return res.status(503).json({
        error: 'OPENAI_API_KEY missing',
        error_ru: 'Нет OPENAI_API_KEY на сервере'
      });
    }
    const user = authUser(req);
    const devPay = String(process.env.ALLOW_DEV_PAYMENTS || '') === '1';
    const plan = (user && user.plan) || 'free';
    const allowed = devPay || plan === 'studio';
    if (!allowed) {
      return res.status(403).json({
        error: 'Studio plan required',
        error_ru: 'Анализ аккаунта доступен на тарифе Studio ($24). Или включите ALLOW_DEV_PAYMENTS=1 для тестов.',
        code: 'PLAN'
      });
    }

    const body = req.body || {};
    const username = String(body.username || '').replace(/^@/, '').trim().slice(0, 64);
    const niche = String(body.niche || '').trim().slice(0, 120);
    const goal = String(body.goal || '').trim().slice(0, 200);
    const bio = String(body.bio || '').trim().slice(0, 500);
    const stats = String(body.stats || '').trim().slice(0, 400);
    const posts = String(body.posts || '').trim().slice(0, 2500);
    const language = body.language === 'en' ? 'en' : (body.language === 'uz' ? 'uz' : 'ru');
    if (!username && !posts && !bio) {
      return res.status(400).json({
        error: 'Provide username or post descriptions',
        error_ru: 'Укажите @username или описание постов'
      });
    }

    const langName = language === 'en' ? 'English' : (language === 'uz' ? 'Uzbek' : 'Russian');
    const system = `You are a senior Instagram growth strategist and short-form content director (Reels-first).
You audit CREATOR ACCOUNTS, not a single video. Be specific, honest, no fake view guarantees.
Output JSON only.
Language of all text fields: ${langName}.

Score dimensions 0-100: positioning, content_pillars, hooks_pattern, posting_rhythm, visual_grid, cta_funnel, differentiation.
Give 6-8 prioritized actions. Each action: what is wrong, what to change this week, why it matters.
Never promise "X followers in Y days". Speak in ranges and experiments.`;

    const userPrompt = `Audit this Instagram account.

Username: @${username || 'unknown'}
Niche: ${niche || 'not specified'}
Goal: ${goal || 'not specified'}
Bio / profile text: ${bio || 'not provided'}
Stats (followers, avg views, ER if any): ${stats || 'not provided'}
Recent posts / Reels notes or links (user-provided):
${posts || 'not provided'}

Return JSON:
{
  "overall_score": 0-100,
  "summary": "5-8 sentences: who this account is for, what already works, main growth bottleneck",
  "scores": {
    "positioning": 0-100,
    "content_pillars": 0-100,
    "hooks_pattern": 0-100,
    "posting_rhythm": 0-100,
    "visual_grid": 0-100,
    "cta_funnel": 0-100,
    "differentiation": 0-100
  },
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "actions": [
    {"priority":"high|mid|low","title":"...","detail":"2-4 sentences concrete"}
  ],
  "7_day_plan": ["day theme or task", "..."],
  "bio_rewrite": "one improved bio option",
  "content_pillars_suggested": ["pillar1","pillar2","pillar3"]
}`;

    const completion = await client().chat.completions.create({
      model: process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini',
      temperature: 0.35,
      max_tokens: 2200,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: userPrompt }
      ]
    });

    const raw = completion.choices?.[0]?.message?.content || '{}';
    let parsed;
    try { parsed = JSON.parse(raw); } catch { parsed = { summary: raw, actions: [] }; }

    if (user) {
      updateUser(user.email, (u) => {
        u.analysesUsed = (u.analysesUsed || 0) + 1;
      });
    }

    res.json({
      ok: true,
      username: username || null,
      overall_score: Number(parsed.overall_score) || 0,
      summary: String(parsed.summary || '').slice(0, 2000),
      scores: parsed.scores || {},
      strengths: Array.isArray(parsed.strengths) ? parsed.strengths.slice(0, 8) : [],
      weaknesses: Array.isArray(parsed.weaknesses) ? parsed.weaknesses.slice(0, 8) : [],
      actions: Array.isArray(parsed.actions) ? parsed.actions.slice(0, 10) : [],
      plan7: Array.isArray(parsed['7_day_plan']) ? parsed['7_day_plan'].slice(0, 10) : (Array.isArray(parsed.plan7) ? parsed.plan7.slice(0, 10) : []),
      bio_rewrite: String(parsed.bio_rewrite || '').slice(0, 400),
      pillars: Array.isArray(parsed.content_pillars_suggested) ? parsed.content_pillars_suggested.slice(0, 6) : [],
      engine: process.env.OPENAI_VISION_MODEL || 'gpt-4o-mini'
    });
  } catch (err) {
    console.error('[account-audit]', err.message);
    res.status(err?.status || 500).json({ error: err.message || 'Account audit failed' });
  }
});


app.get('*', (req, res, next) => {
  if (req.path.startsWith('/api')) return next();
  const index = path.join(publicDir, 'index.html');
  if (fs.existsSync(index)) res.sendFile(index);
  else res.status(404).send('Frontend not found');
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`ViralClip http://localhost:${PORT} (also LAN via your PC IP)`);
  console.log(
    hasKey()
      ? 'OpenAI key: configured'
      : 'OpenAI key: MISSING — set OPENAI_API_KEY in .env'
  );
  console.log(
    `Limits: STT ${LIMIT_TRANSCRIBE_PER_HOUR}/h, AI ${LIMIT_ANALYZE_PER_HOUR}/h, ${LIMIT_ANALYZE_PER_DAY}/day`
  );
});
