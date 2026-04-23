# Design Document: Hebrew Writing Coach — Web Application

## 1. Product Vision

A web application that analyzes Hebrew text in real time, surfaces linguistic diagnoses, and guides users through targeted revisions with measurable improvement. The app uses the ML distillation model (Layer 6) as its analysis engine and Amazon Bedrock LLMs for AI-powered rewrite suggestions.

### Core Loop

```
Write → Analyze → Diagnose → Suggest → Revise (with LLM help) → Re-analyze → Show Progress
```

### Target Audience

Regular Hebrew writers: students, bloggers, professionals writing reports, non-native speakers. The entire UI is in Hebrew.

### Key Decisions

- **Full Hebrew UI** — all text, buttons, navigation, labels in Hebrew
- **No user accounts** — anonymous usage, admin pages password-protected
- **Mobile-friendly** — responsive design that works on phones
- **LLM rewrite suggestions from Phase 1** — via Amazon Bedrock
- **Deployment on AWS** — ECS and EKS
- **Example texts** — users can load pre-built examples (tweets, news articles, legal text, blog posts)
- **Spider chart for scores** — 5 linguistic scores visualized as a radar/spider chart (recharts) rather than individual bars, with before/after overlay during revision comparison
- **Methodology page** — public `/methodology` route explaining the analysis system in Hebrew (features, scores, issues, diagnoses, interventions) with collapsible sections
- **Frontend-design skill** — all frontend components implemented using the `frontend-design` skill for production-grade UI quality

## 2. UX Architecture

### 2.1 Primary Screen — Editor + Analysis Panel

Two-pane layout (stacks vertically on mobile):

```
┌─────────────────────────────┬──────────────────────────┐
│                             │  ציונים                   │
│   עורך טקסט                 │  ████████░░ קושי      70 │
│   (Monaco / RTL עברית)      │  ███░░░░░░░ סגנון     31 │
│                             │  ████░░░░░░ שטף       39 │
│   המשתמש מקליד כאן...       │  █████░░░░░ קוהרנטיות 51 │
│                             │  ██████░░░░ מורכבות   65 │
│   משפטים מודגשים            │                          │
│   לפי חומרה                 │  אבחנות                   │
│                             │  ⚠ קוהרנטיות נמוכה (0.79)│
│  ┌─────────────────────┐    │  ⚠ שימוש יתר בכינויים    │
│  │ 📂 דוגמאות          │    │                          │
│  │ ציוץ | חדשות | משפטי│    │  המלצות                   │
│  └─────────────────────┘    │  📝 הוסיפו מילות קישור... │
│                             │  📝 החליפו כינויי גוף... │
└─────────────────────────────┴──────────────────────────┘
```

**Behavior:**
1. User types, pastes, or loads an example text
2. After 800ms debounce, call the model
3. Update analysis panel and inline highlights
4. RTL text direction throughout

### 2.2 Example Text Selector

A dropdown or button bar at the top of the editor lets users load pre-built examples:

| Category | Label (Hebrew) | Description |
|----------|----------------|-------------|
| Tweet | ציוץ | Financial blogger tweets (sample3, sample4 style) |
| News | כתבה חדשותית | News article (sample1 style) |
| Legal | טקסט משפטי | Legal/government text (sample2 style) |
| Essay | חיבור | Student essay example |
| Blog | פוסט בבלוג | Blog post example |

Examples are stored as static JSON files on the server with pre-computed analysis results for instant display. When the user selects one, the text loads into the editor and analysis runs immediately.

### 2.3 Inline Feedback — Sentence Highlighting

Overlay annotations directly in the editor:

- **Red background** → high complexity severity (> 0.7)
- **Yellow background** → medium severity (0.4–0.7)
- **Red wavy underline** between sentence pairs with weak cohesion (> 0.5)

**Hover tooltips (in Hebrew):**
- Issue type name
- Short explanation
- Suggested action

### 2.4 Intervention Cards

Each active diagnosis becomes an actionable card:

```
┌─────────────────────────────────┐
│ ⚠ קוהרנטיות נמוכה              │
│ חומרה: ████████░░ 0.79          │
│                                 │
│ למה: המשפטים מחוברים בצורה     │
│ חלשה זה לזה                     │
│                                 │
│ מה לעשות:                       │
│ • הוסיפו מילות קישור            │
│ • שפרו את הזרימה בין משפטים     │
│                                 │
│ [🔧 תרגול: שכתוב עם מקשרים]    │
│ [🤖 הצעת שכתוב מ-AI]           │
└─────────────────────────────────┘
```

Two action buttons per card:
- **תרגול (Practice)** → manual guided rewrite mode
- **הצעת שכתוב מ-AI** → LLM generates a rewrite suggestion

### 2.5 Guided Rewrite Mode

When the user clicks Practice or AI Rewrite:

1. The relevant sentence(s) are highlighted and isolated
2. An editable text box appears below the original
3. If AI rewrite: the LLM suggestion pre-fills the edit box (user can modify)
4. Real-time re-scoring shows the impact

```
┌─────────────────────────────────┐
│ מקור:                           │
│ "הוא הלך. הוא ראה. הוא חזר."   │
│                                 │
│ הצעת שכתוב:                     │
│ ┌─────────────────────────────┐ │
│ │ הוא הלך, ולאחר שראה את     │ │
│ │ המקום, חזר הביתה.           │ │
│ └─────────────────────────────┘ │
│                                 │
│ שינויים:                        │
│ -הוא הלך. הוא ראה. הוא חזר.   │
│ +הוא הלך, ולאחר שראה את       │
│ +המקום, חזר הביתה.             │
│                                 │
│ קוהרנטיות: 0.42 → 0.71 ↑ (+69%)│
│ מורכבות:   0.31 → 0.55 ↑       │
│                                 │
│ [✓ החל] [✗ בטל]                │
└─────────────────────────────────┘
```

The diff view shows added text in green, removed in red — so the user understands exactly what changed.

### 2.6 Progress Feedback

After each revision:

```
┌─────────────────────────────────┐
│ 📊 השפעת העריכה                 │
│                                 │
│ קוהרנטיות: 0.42 → 0.61  ↑ +45% │
│ שטף:       0.55 → 0.68  ↑ +24% │
│ מורכבות:   0.65 → 0.58  ↓ -11% │
│                                 │
│ אבחנות שנפתרו: 1 מתוך 3       │
│ ✅ קוהרנטיות נמוכה — תוקן!     │
└─────────────────────────────────┘
```

### 2.7 Export / Share

Users can export their analysis as:
- **PDF report** — text + scores + diagnoses + interventions
- **Share link** — generates a URL with the text encoded (no server storage needed for MVP — base64 in URL or short-lived server cache)

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Frontend   │────▶│   API Layer  │────▶│  ML Model Service│
│   (React)    │◀────│   (FastAPI)  │◀────│  (PyTorch)       │
└──────────────┘     └──────────────┘     └──────────────────┘
                            │                      │
                            │                      ▼ (fallback)
                            │              ┌──────────────────┐
                            │              │  Rule Pipeline   │
                            │              └──────────────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │  Amazon Bedrock  │
                     │  (LLM Rewrite)   │
                     └──────────────────┘
```

### 3.2 AWS Deployment

```
Route 53 (DNS)
    │
    ▼
ALB (Application Load Balancer)
    │
    ├── /           → ECS Service: Frontend (Nginx + React SPA)
    ├── /api/*      → ECS Service: Backend (FastAPI + ML model)
    ├── /api/rewrite → Backend → Amazon Bedrock
    └── /admin/*    → Backend (password-protected)
```

**ECS Services:**
- **Frontend:** Nginx container serving the React build. Lightweight, scales horizontally.
- **Backend:** FastAPI container with the ML model loaded in memory (~740MB). GPU instance (g4dn.xlarge) for fast inference, or CPU (m5.xlarge) for cost savings.

**EKS** for production scaling: Kubernetes manages auto-scaling, rolling deployments, and health checks. ECS for MVP simplicity, migrate to EKS when scaling demands it.

### 3.3 Amazon Bedrock Integration

The LLM rewrite feature uses Amazon Bedrock to generate Hebrew text suggestions.

**Admin-configurable LLM selection:**
- Admin page allows choosing the Bedrock model (Claude Sonnet 4, Claude Opus 4, Nova Pro, etc.)
- Model selection stored in server config (environment variable or config file)
- No user-facing model choice — admin decides which model to use

**Rewrite prompt template:**
```
You are an expert Hebrew writing coach. The user's text has a specific linguistic issue that needs fixing.

Issue diagnosed: {diagnosis_label_he}
Why it's a problem: {explanation_he}

Specific actions to take:
{actions_list}

Quick tip: {tip_he}

Original text:
{original_text}

Rewrite the text applying the specific actions listed above. Fix ONLY the diagnosed issue — keep the meaning, tone, and content the same. The result should be natural Hebrew prose.
Respond only with the rewritten Hebrew text, no explanations or commentary.
```

**API flow:**
```
User clicks "הצעת שכתוב מ-AI"
    → Frontend sends POST /api/rewrite
    → Backend builds prompt from diagnosis + sentence
    → Backend calls Bedrock InvokeModel
    → Returns rewritten text to frontend
    → User sees suggestion in the rewrite modal
```

## 4. API Design

### POST /api/analyze

Analyzes text and returns full diagnostic output.

**Request:**
```json
{
  "text": "Hebrew text here..."
}
```

**Response:**
```json
{
  "scores": {
    "difficulty": 0.703,
    "style": 0.137,
    "fluency": 0.469,
    "cohesion": 0.506,
    "complexity": 0.653
  },
  "diagnoses": [
    {
      "type": "low_cohesion",
      "severity": 0.606,
      "label_he": "קוהרנטיות נמוכה",
      "explanation_he": "המשפטים מחוברים בצורה חלשה זה לזה",
      "actions_he": ["הוסיפו מילות קישור בין משפטים", "שפרו את הזרימה"],
      "tip_he": "נסו להוסיף מילות קישור כמו 'לכן', 'בנוסף', 'עם זאת'"
    }
  ],
  "interventions": [
    {
      "type": "cohesion_improvement",
      "priority": 0.606,
      "target_diagnosis": "low_cohesion",
      "actions_he": ["הוסיפו מילות קישור בין משפטים"],
      "exercises_he": ["הכניסו מקשרים מתאימים לקטע"]
    }
  ],
  "sentences": [
    {"index": 0, "text": "...", "char_start": 0, "char_end": 45, "complexity": 0.515, "highlight": "yellow"},
    {"index": 1, "text": "...", "char_start": 46, "char_end": 120, "complexity": 0.839, "highlight": "red"}
  ],
  "cohesion_gaps": [
    {"pair": [0, 1], "severity": 0.498, "char_start": 45, "char_end": 46}
  ]
}
```

### POST /api/revise

Compares original and edited text.

**Request:**
```json
{
  "original_text": "Original Hebrew text...",
  "edited_text": "Revised Hebrew text..."
}
```

**Response:**
```json
{
  "original_scores": {"difficulty": 0.70, "cohesion": 0.42},
  "revised_scores": {"difficulty": 0.65, "cohesion": 0.61},
  "deltas": {"difficulty": -0.05, "cohesion": +0.19},
  "resolved_diagnoses": ["low_cohesion"],
  "new_diagnoses": []
}
```

### POST /api/rewrite

Generates an LLM rewrite suggestion for a specific diagnosis.

**Request:**
```json
{
  "text": "The sentence(s) to rewrite...",
  "diagnosis_type": "low_cohesion",
  "context": "Optional surrounding text for context..."
}
```

**Response:**
```json
{
  "suggestion": "Rewritten Hebrew text...",
  "model_used": "us.anthropic.claude-sonnet-4-20250514-v1:0"
}
```

### POST /api/exercise

Generates a multiple-choice rewrite exercise for a diagnosed issue. Uses Amazon Bedrock to produce 3 rewrite options: one correct fix and two with different problems, each with a Hebrew explanation.

**Request:**
```json
{
  "text": "Problematic sentence(s)...",
  "diagnosis_type": "low_cohesion"
}
```

**Response:**
```json
{
  "original_text": "Problematic sentence(s)...",
  "diagnosis_label_he": "קוהרנטיות נמוכה",
  "diagnosis_explanation_he": "המשפטים מחוברים בצורה חלשה זה לזה",
  "tip_he": "הוסיפו מילות קישור כמו 'לכן', 'בנוסף'",
  "options": [
    {"text": "Rewritten option 1...", "is_correct": false, "explanation_he": "..."},
    {"text": "Rewritten option 2...", "is_correct": true, "explanation_he": "..."},
    {"text": "Rewritten option 3...", "is_correct": false, "explanation_he": "..."}
  ]
}
```

Returns 400 if the diagnosis type is not one of the 8 recognized types. Returns 503 if Bedrock is unavailable or the LLM response cannot be parsed.

### GET /api/examples

Returns the list of available example texts.

**Response:**
```json
{
  "examples": [
    {"id": "tweet_finance", "label": "ציוץ פיננסי", "category": "ציוץ", "preview": "אלה היו 15 השנים..."},
    {"id": "news_security", "label": "כתבה ביטחונית", "category": "חדשות", "preview": "האקרים של המודיעין..."},
    {"id": "legal_pension", "label": "טקסט משפטי — פנסיה", "category": "משפטי", "preview": "במקרה שהעובד..."}
  ]
}
```

### GET /api/examples/{id}

Returns the full text of an example.

### Admin Endpoints (password-protected)

### GET /admin/config

Returns current admin settings.

### POST /admin/config

Updates admin settings.

```json
{
  "bedrock_model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  "severity_threshold": 0.3,
  "max_diagnoses_shown": 3,
  "max_interventions_shown": 3
}
```

### GET /admin/models

Lists available Bedrock models.

## 5. Frontend Architecture

### 5.1 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Framework | React 18 + TypeScript | Industry standard |
| Editor | Monaco Editor | RTL support, extensible highlighting |
| State | Zustand | Lightweight |
| Styling | Tailwind CSS | Rapid development, RTL utilities |
| HTTP | Axios | Simple API calls |
| Build | Vite | Fast dev server |
| Diff | diff-match-patch | Inline diff view for rewrites |
| PDF | jsPDF | Export to PDF |

### 5.2 Component Tree

```
App (dir="rtl" lang="he")
├── Header
│   ├── Logo + Title ("מאמן כתיבה בעברית")
│   └── ExportButton ("ייצוא" / "שיתוף")
├── ExampleSelector
│   └── ExampleButton × N
├── MainLayout (two-pane, stacks on mobile)
│   ├── EditorPane
│   │   ├── MonacoEditor (RTL)
│   │   └── InlineAnnotations
│   └── AnalysisPanel
│       ├── ScoreGauges (5 gauges)
│       ├── DiagnosisList
│       │   └── DiagnosisCard × N
│       ├── InterventionList
│       │   └── InterventionCard × N
│       └── ProgressFeedback
├── RewriteModal
│   ├── OriginalText
│   ├── RewriteEditor
│   ├── DiffView
│   └── DeltaScores
└── AdminPage (password-protected route)
    ├── ModelSelector (Bedrock models)
    ├── ThresholdSlider
    └── DisplaySettings
```

### 5.3 Mobile Layout

On screens < 768px:
- Panes stack vertically (editor on top, analysis below)
- Analysis panel becomes a collapsible accordion
- Intervention cards become full-width
- Rewrite modal becomes full-screen

## 6. Backend Architecture

### 6.1 Project Structure

```
app/
├── main.py                  # FastAPI app, CORS, startup
├── api/
│   ├── analyze.py           # POST /api/analyze
│   ├── revise.py            # POST /api/revise
│   ├── rewrite.py           # POST /api/rewrite (Bedrock LLM)
│   ├── examples.py          # GET /api/examples
│   ├── admin.py             # Admin endpoints (password-protected)
│   └── health.py            # GET /api/health
├── services/
│   ├── model_service.py     # ML model loading + inference
│   ├── bedrock_service.py   # Amazon Bedrock LLM integration
│   ├── localization.py      # Hebrew labels, explanations, tips
│   └── example_service.py   # Example text management
├── models/
│   └── schemas.py           # Pydantic request/response models
├── data/
│   └── examples/            # Example text JSON files
├── config.py                # Settings (model path, Bedrock config, admin password)
└── Dockerfile
```

### 6.2 Bedrock Service

```python
class BedrockService:
    def __init__(self, region: str = "eu-west-1"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = settings.BEDROCK_MODEL_ID
    
    def rewrite(self, text: str, diagnosis_type: str, context: str = "") -> str:
        """Generate a rewrite suggestion using the configured Bedrock model."""
        prompt = self._build_prompt(text, diagnosis_type, context)
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"messages": [{"role": "user", "content": prompt}]}),
        )
        return self._parse_response(response)
    
    def list_models(self) -> list[ModelInfo]:
        """List available Bedrock foundation models for text generation.

        Returns a static curated list of Anthropic and Amazon models
        (Claude 4.x, Nova).  No Bedrock API call is made —
        the list is maintained in code.
        """
        ...
```

### 6.3 Admin Authentication

Simple password-based protection for admin endpoints:

```python
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "change-me")

def verify_admin(password: str = Header(..., alias="X-Admin-Password")):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
```

No user accounts, no database, no sessions. Admin password set via environment variable.

## 7. Hebrew Localization

All 8 diagnosis types with Hebrew labels, explanations, and tips:

| Diagnosis | Hebrew Label | Explanation | Tip |
|-----------|-------------|-------------|-----|
| low_cohesion | קוהרנטיות נמוכה | המשפטים מחוברים בצורה חלשה | הוסיפו מילות קישור כמו 'לכן', 'בנוסף' |
| sentence_over_complexity | משפטים מורכבים מדי | חלק מהמשפטים ארוכים ומסובכים | פצלו משפטים ארוכים לשניים-שלושה קצרים |
| low_lexical_diversity | אוצר מילים מצומצם | יש חזרה רבה על אותן מילים | החליפו מילים חוזרות במילים נרדפות |
| pronoun_overuse | שימוש יתר בכינויי גוף | יותר מדי 'הוא', 'היא', 'הם' | החליפו כינויים בשמות מפורשים |
| structural_inconsistency | חוסר עקביות מבנית | המשפטים שונים מאוד במבנה | שמרו על אורך ומבנה דומים |
| low_morphological_richness | מגוון מורפולוגי נמוך | שימוש מצומצם בצורות פועל | השתמשו בבניינים שונים |
| fragmented_writing | כתיבה מקוטעת | יותר מדי משפטים קצרים | חברו משפטים קצרים למשפטים שלמים |
| punctuation_deficiency | בעיות פיסוק | חסרים סימני פיסוק | בדקו נקודות בסוף משפטים ופסיקים |

Score labels:

| Score | Hebrew |
|-------|--------|
| difficulty | קושי |
| style | סגנון |
| fluency | שטף |
| cohesion | קוהרנטיות |
| complexity | מורכבות |

## 8. Admin Panel

Password-protected page at `/admin`:

### Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| Bedrock Model | Dropdown | Claude Sonnet 4.5 | LLM for rewrite suggestions |
| Severity Threshold | Slider 0.0–1.0 | 0.3 | Minimum severity to show a diagnosis |
| Max Diagnoses | Number | 3 | Maximum diagnoses shown to user |
| Max Interventions | Number | 3 | Maximum interventions shown |

### Model Management

- List available Bedrock models
- Select active model
- Test rewrite with sample text

## 9. Performance Targets

| Action | Target | How |
|--------|--------|-----|
| Typing feedback (debounce) | < 800ms | Debounce timer |
| Model inference | < 500ms | GPU inference, model in memory |
| UI update after analysis | < 100ms | React state update |
| LLM rewrite suggestion | < 3s | Bedrock streaming |
| Rewrite comparison | < 700ms | Two model inference calls |
| Page load | < 2s | Static SPA, model pre-loaded |

## 10. MVP Scope

### Phase 1 — Core App (2–3 weeks)

- [ ] FastAPI backend with `/api/analyze`, `/api/revise`, `/api/rewrite` endpoints
- [ ] ML model loaded in memory at startup
- [ ] Amazon Bedrock integration for LLM rewrite suggestions
- [ ] React frontend with Monaco Editor (RTL Hebrew)
- [ ] Full Hebrew UI
- [ ] Analysis panel: 5 scores, top 3 diagnoses, top 3 interventions
- [ ] Sentence highlighting (red/yellow by complexity)
- [ ] Cohesion gap markers
- [ ] Intervention cards with Practice + AI Rewrite buttons
- [ ] Guided rewrite modal with diff view and delta scores
- [ ] Example text selector (5 pre-built examples)
- [ ] Export to PDF
- [ ] Admin page (password-protected): model selection, severity threshold
- [ ] Mobile-friendly responsive layout
- [ ] Docker container for local development
- [ ] ECS deployment with ALB

### Phase 2 — Polish & Scale (2–3 weeks)

- [ ] Share link generation
- [ ] Progress feedback with before/after visualization
- [ ] More example texts (10+)
- [ ] Bedrock streaming for faster rewrite display
- [ ] EKS migration for auto-scaling
- [ ] CloudFront CDN for static assets
- [ ] Monitoring and logging (CloudWatch)

### Phase 3 — Personalization (2–3 weeks)

- [ ] Session history (localStorage)
- [ ] Frequent diagnosis tracking
- [ ] Gamification: improvement streaks, skill bars
- [ ] Additional LLM features: explain why a rewrite is better
- [ ] A/B testing different LLM models for rewrite quality

## 11. Deployment

### Docker Compose (Local Development)

```yaml
services:
  frontend:
    build: ./frontend
    ports: ["3000:80"]
  
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - MODEL_PATH=/models/model_v5
      - AWS_REGION=eu-west-1
      - ADMIN_PASSWORD=dev-password
    volumes:
      - ./model_v5:/models/model_v5
```

### ECS (Production)

```
ECS Cluster
├── Service: frontend
│   ├── Task: nginx + React SPA
│   ├── CPU: 256, Memory: 512MB
│   └── Desired count: 2
│
└── Service: backend
    ├── Task: FastAPI + ML model
    ├── CPU: 2048, Memory: 4096MB (or GPU)
    └── Desired count: 1–3 (auto-scaling)
```

### Infrastructure (Terraform/CDK)

- VPC with public/private subnets
- ALB with HTTPS (ACM certificate)
- ECS cluster with Fargate or EC2 launch type
- ECR for Docker images
- S3 for static assets (optional, or serve from Nginx)
- CloudWatch for logs and metrics
- IAM roles for Bedrock access

## 12. Key Design Decisions

1. **No user accounts.** Anonymous usage. Admin pages password-protected via environment variable. No database needed for MVP.

2. **LLM from Phase 1.** Bedrock integration is core to the product — the AI rewrite suggestions are the differentiator. Admin controls which model is used. Default model: Claude Sonnet 4.5 (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`).

3. **Full Hebrew UI.** Every string the user sees is in Hebrew. English only in code and API internals.

4. **Mobile-friendly.** Responsive design from day one. Two-pane layout stacks on mobile.

5. **Stateless model.** Each API call is independent. No session state on the server.

6. **Admin-controlled severity threshold.** The admin decides how sensitive the analysis is — users see a curated experience.

7. **Example texts for onboarding.** New users can immediately see the app in action by loading a pre-built example.

8. **Diff view in rewrites.** Users see exactly what changed (green/red inline diff) so they learn from each edit.

9. **Spider chart for scores.** The 5 linguistic scores are visualized as a radar/spider chart (recharts RadarChart) rather than individual bars. This gives users an at-a-glance profile of their text's strengths and weaknesses. The chart supports before/after overlay during revision comparison.

10. **Methodology page.** A public `/methodology` route explains in Hebrew how the system works — features, scores, issues, diagnoses, and interventions — building user trust and helping learners understand the feedback.

11. **Frontend-design skill.** All frontend components are implemented using the `frontend-design` skill to ensure production-grade, distinctive UI with high design quality.

## 13. Actual API Response Examples

### POST /api/analyze — Example Response

```json
{
  "scores": {
    "difficulty": 0.703,
    "style": 0.137,
    "fluency": 0.469,
    "cohesion": 0.506,
    "complexity": 0.653
  },
  "diagnoses": [
    {
      "type": "low_cohesion",
      "severity": 0.606,
      "label_he": "קוהרנטיות נמוכה",
      "explanation_he": "המשפטים מחוברים בצורה חלשה זה לזה",
      "actions_he": ["הוסיפו מילות קישור בין משפטים", "שפרו את הזרימה"],
      "tip_he": "נסו להוסיף מילות קישור כמו 'לכן', 'בנוסף', 'עם זאת'"
    },
    {
      "type": "sentence_over_complexity",
      "severity": 0.523,
      "label_he": "משפטים מורכבים מדי",
      "explanation_he": "חלק מהמשפטים ארוכים ומסובכים מדי",
      "actions_he": ["פצלו משפטים ארוכים", "הפחיתו שימוש בפסוקיות משנה"],
      "tip_he": "פצלו משפטים ארוכים לשניים-שלושה קצרים"
    }
  ],
  "interventions": [
    {
      "type": "cohesion_improvement",
      "priority": 0.606,
      "target_diagnosis": "low_cohesion",
      "actions_he": ["הוסיפו מילות קישור בין משפטים"],
      "exercises_he": ["הכניסו מקשרים מתאימים לקטע"]
    },
    {
      "type": "sentence_simplification",
      "priority": 0.523,
      "target_diagnosis": "sentence_over_complexity",
      "actions_he": ["פצלו משפטים ארוכים"],
      "exercises_he": ["שכתבו משפטים מעל 30 מילים"]
    }
  ],
  "sentences": [
    {"index": 0, "text": "הילד הלך לבית הספר.", "char_start": 0, "char_end": 19, "complexity": 0.315, "highlight": "none"},
    {"index": 1, "text": "הוא למד שם כל היום ולאחר מכן חזר הביתה עם חבריו.", "char_start": 20, "char_end": 70, "complexity": 0.652, "highlight": "yellow"},
    {"index": 2, "text": "בערב הוא ישב ללמוד.", "char_start": 71, "char_end": 90, "complexity": 0.285, "highlight": "none"}
  ],
  "cohesion_gaps": [
    {"pair": [0, 1], "severity": 0.498, "char_start": 19, "char_end": 20}
  ]
}
```

### POST /api/revise — Example Response

```json
{
  "original_scores": {"difficulty": 0.703, "style": 0.137, "fluency": 0.469, "cohesion": 0.506, "complexity": 0.653},
  "revised_scores": {"difficulty": 0.651, "style": 0.185, "fluency": 0.512, "cohesion": 0.614, "complexity": 0.598},
  "deltas": {"difficulty": -0.052, "style": 0.048, "fluency": 0.043, "cohesion": 0.108, "complexity": -0.055},
  "resolved_diagnoses": ["low_cohesion"],
  "new_diagnoses": []
}
```

### POST /api/rewrite — Example Response

```json
{
  "suggestion": "הילד הלך לבית הספר, ולאחר שלמד שם כל היום, חזר הביתה עם חבריו. בערב הוא ישב ללמוד.",
  "model_used": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
}
```

### GET /api/examples — Example Response

```json
{
  "examples": [
    {"id": "tweet_finance", "label": "ציוץ פיננסי", "category": "ציוץ", "preview": "אלה היו 15 השנים..."},
    {"id": "news_security", "label": "כתבה ביטחונית", "category": "כתבה חדשותית", "preview": "האקרים של המודיעין..."},
    {"id": "legal_pension", "label": "טקסט משפטי — פנסיה", "category": "טקסט משפטי", "preview": "במקרה שהעובד..."},
    {"id": "essay_education", "label": "חיבור — חינוך", "category": "חיבור", "preview": "מערכת החינוך..."},
    {"id": "blog_tech", "label": "פוסט בבלוג — טכנולוגיה", "category": "פוסט בבלוג", "preview": "בשנים האחרונות..."}
  ]
}
```

### GET /api/health — Example Responses

Healthy:
```json
{"status": "healthy", "model_loaded": true}
```

Unhealthy (503):
```json
{"status": "unhealthy", "model_loaded": false}
```

## 14. Methodology Page Content Structure

The methodology page (`/methodology`) is a public page explaining the analysis system in Hebrew. It uses collapsible sections so users can expand only the topics they are interested in. The page is accessible via a "על המתודולוגיה" link in the Analysis Panel.

### Section 1: תכונות (Features) — Default Open

Introductory text: "המערכת מחלצת למעלה מ-30 תכונות לשוניות מהטקסט, מחולקות לשש קטגוריות."

Six feature categories, each with a colored header and a list of features with Hebrew descriptions:

| Category | Hebrew | Features |
|----------|--------|----------|
| Morphology | מורפולוגיה | verb_ratio, binyan_distribution, prefix_density, suffix_pronoun_ratio, morphological_ambiguity, agreement_error_rate, binyan_entropy, construct_ratio |
| Syntax | תחביר | avg_sentence_length, avg_tree_depth, max_tree_depth, avg_dependency_distance, clauses_per_sentence, subordinate_clause_ratio, right_branching_ratio, dependency_distance_variance |
| Lexicon | לקסיקון | type_token_ratio, hapax_ratio, avg_token_length, lemma_diversity, rare_word_ratio, content_word_ratio |
| Structure | מבנה | sentence_length_variance, long_sentence_ratio, punctuation_ratio, short_sentence_ratio, missing_terminal_punctuation_ratio |
| Discourse | שיח | connective_ratio, sentence_overlap, pronoun_to_noun_ratio |
| Style | סגנון | sentence_length_trend, pos_distribution_variance |

### Section 2: ציונים (Scores)

Introductory text: "חמישה ציונים מורכבים מחושבים מהתכונות. כל ציון הוא שילוב משוקלל של ערכי תכונות מנורמלים."

Each score displayed as a card with:
- Hebrew name and range badge (0.0–1.0)
- Hebrew description of what the score measures
- Formula in plain Hebrew (e.g., "שילוב משוקלל של אורך משפטים, עומק עצי תחביר, יחס מילים נדירות, ועמימות מורפולוגית")

Scores: קושי (difficulty), סגנון (style), שטף (fluency), קוהרנטיות (cohesion), מורכבות (complexity).

### Section 3: בעיות (Issues)

Introductory text: "17 סוגי בעיות ב-6 קבוצות. כל בעיה מזוהה באמצעות ציון רך (sigmoid) המבוסס על סטיית ערך התכונה מממוצע הקורפוס."

Issues grouped by category with Hebrew names and severity formula descriptions.

### Section 4: אבחנות (Diagnoses)

Introductory text: "8 כללי אבחון מצרפים דפוסי בעיות וציונים מורכבים לאבחנות לשוניות משמעותיות."

Each diagnosis displayed as a card with:
- Hebrew name
- Activation threshold badge
- Formula description in Hebrew

### Section 5: המלצות (Interventions)

Introductory text: "4 סוגי התערבות פדגוגית ממופים מהאבחנות. כל אבחנה ממופה לסוג התערבות אחד בדיוק."

Each intervention displayed as a card with:
- Hebrew name
- Hebrew description
- Trigger diagnosis tags (which diagnoses activate this intervention)

### Footer

"המתודולוגיה מבוססת על קורפוס ויקיפדיה עברית ומודל DictaBERT מזוקק"

## 15. Implementation Notes

### Changes from Original Design

1. **Default Bedrock model** — Changed to Claude Sonnet 4.5 (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`). Admin can still select any available Bedrock model. Curated model IDs use cross-region inference profile format with a prefix derived from `AWS_REGION`: `eu` for EU regions, `ap` for Asia-Pacific, `us` otherwise.
2. **PDF export** — Uses jsPDF (not html2pdf.js) for PDF generation.
3. **Score visualization** — Spider/radar chart (recharts RadarChart) replaces the original individual score bars design. Provides better at-a-glance comparison of text strengths and weaknesses.
4. **Methodology page** — Added as a new feature not in the original MVP scope. Provides transparency into the analysis system.
5. **Frontend-design skill** — All frontend components use the `frontend-design` skill for production-grade UI quality, resulting in a custom design system with CSS variables for theming.
6. **Inline annotations** — Implemented as a custom React hook (`useInlineAnnotations`) using Monaco's `deltaDecorations` API rather than a separate component.
7. **Home page** — The main editor + analysis view is implemented as a `HomePage` component at the `/` route, with `MainLayout` handling the two-pane responsive layout.
