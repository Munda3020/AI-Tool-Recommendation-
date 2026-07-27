"""
STEP 1: Build the full catalog.
110+ real AI tools active in 2026, across 20 categories - researched from
current 2026 industry roundups, not guessed. This is the backbone of the
whole project: everything downstream (SQL analysis, ML classifier, search,
recommender) depends on this being a genuinely researched dataset, not a
toy list of 10 famous names.
"""
import pandas as pd

tools = [
    # --- General AI Assistants ---
    {"tool_name": "ChatGPT", "category": "General AI Assistant", "pricing_tier": "Freemium",
     "best_for": "Most versatile all-purpose assistant; writing, research, coding, voice",
     "tags": "chat assistant writing general knowledge coding voice multimodal versatile"},
    {"tool_name": "Claude", "category": "General AI Assistant", "pricing_tier": "Freemium",
     "best_for": "Long-form writing, careful reasoning, and large codebase understanding",
     "tags": "chat assistant writing reasoning coding long context analysis careful"},
    {"tool_name": "Gemini", "category": "General AI Assistant", "pricing_tier": "Freemium",
     "best_for": "Deep Google Workspace integration and large multimodal context",
     "tags": "chat assistant google workspace multimodal video image research"},
    {"tool_name": "Grok", "category": "General AI Assistant", "pricing_tier": "Freemium",
     "best_for": "Real-time fact-checking and X/Twitter-integrated search",
     "tags": "chat assistant fact check real-time social media search"},
    {"tool_name": "Microsoft Copilot", "category": "General AI Assistant", "pricing_tier": "Freemium",
     "best_for": "AI embedded across Word, Excel, Outlook, and Teams",
     "tags": "chat assistant office word excel outlook teams enterprise"},

    # --- Coding & Development ---
    {"tool_name": "Cursor", "category": "Coding & Development", "pricing_tier": "Freemium",
     "best_for": "AI-native code editor for professional, codebase-aware development",
     "tags": "coding editor ide programming refactor codebase autocomplete developer"},
    {"tool_name": "Claude Code", "category": "Coding & Development", "pricing_tier": "Paid",
     "best_for": "Terminal-based agentic coding for complex, multi-file work",
     "tags": "coding agentic terminal debugging refactor programming automation pipeline"},
    {"tool_name": "GitHub Copilot", "category": "Coding & Development", "pricing_tier": "Freemium",
     "best_for": "In-editor code completion tightly integrated with the GitHub ecosystem",
     "tags": "coding autocomplete ide programming developer completion github"},
    {"tool_name": "Replit AI", "category": "Coding & Development", "pricing_tier": "Freemium",
     "best_for": "Browser-based app building and deployment for beginners",
     "tags": "coding app building deployment beginner cloud prototyping browser"},
    {"tool_name": "Aider", "category": "Coding & Development", "pricing_tier": "Free",
     "best_for": "Open-source terminal pair programmer with automatic git commits",
     "tags": "coding terminal git open source pair programming commit"},
    {"tool_name": "Windsurf", "category": "Coding & Development", "pricing_tier": "Freemium",
     "best_for": "AI-native IDE with fast completions for everyday coding",
     "tags": "coding ide editor completion programming developer"},
    {"tool_name": "Lovable", "category": "Coding & Development", "pricing_tier": "Freemium",
     "best_for": "Building full web apps from plain-language prompts, no coding needed",
     "tags": "coding no-code app building website prompt beginner mvp"},
    {"tool_name": "Bolt.new", "category": "Coding & Development", "pricing_tier": "Freemium",
     "best_for": "Instant full-stack app scaffolding and shipping in the browser",
     "tags": "coding no-code app building website scaffolding fast prototyping"},
    {"tool_name": "v0", "category": "Coding & Development", "pricing_tier": "Freemium",
     "best_for": "Generating production-ready UI components from text prompts",
     "tags": "coding ui components frontend design generation react"},

    # --- Search & Research ---
    {"tool_name": "Perplexity", "category": "Search & Research", "pricing_tier": "Freemium",
     "best_for": "Real-time web search with cited, trustworthy summarized answers",
     "tags": "search research citations real-time answers fact finding sources"},
    {"tool_name": "NotebookLM", "category": "Search & Research", "pricing_tier": "Free",
     "best_for": "Source-grounded research answers only from documents you upload",
     "tags": "research documents grounded notes study briefs uploaded files"},
    {"tool_name": "Consensus", "category": "Search & Research", "pricing_tier": "Freemium",
     "best_for": "Finding and summarizing scientific research consensus on a question",
     "tags": "research scientific papers evidence consensus academic search"},
    {"tool_name": "Elicit", "category": "Search & Research", "pricing_tier": "Freemium",
     "best_for": "Academic literature review and research paper summarization",
     "tags": "research academic papers literature review citations systematic review"},
    {"tool_name": "You.com", "category": "Search & Research", "pricing_tier": "Freemium",
     "best_for": "AI-powered search with specialized, app-like modes",
     "tags": "search research web browsing answers"},

    # --- Presentations ---
    {"tool_name": "Gamma", "category": "Presentations", "pricing_tier": "Freemium",
     "best_for": "Fastest prompt-to-deck generation with clean export",
     "tags": "presentation slides deck pitch design generate outline document webpage"},
    {"tool_name": "Beautiful.ai", "category": "Presentations", "pricing_tier": "Paid",
     "best_for": "Auto-formatted slide layouts with zero manual design work",
     "tags": "presentation slides templates auto-design business deck"},
    {"tool_name": "Tome", "category": "Presentations", "pricing_tier": "Freemium",
     "best_for": "Narrative-style AI presentations and pitch decks",
     "tags": "presentation slides storytelling pitch deck narrative"},
    {"tool_name": "Decktopus", "category": "Presentations", "pricing_tier": "Freemium",
     "best_for": "Fast, simple slide generation from a short prompt",
     "tags": "presentation slides quick deck generator"},

    # --- Image Generation ---
    {"tool_name": "Midjourney", "category": "Image Generation", "pricing_tier": "Paid",
     "best_for": "The most artistically impactful, stylized image generation",
     "tags": "image generation art illustration design creative visuals"},
    {"tool_name": "Adobe Firefly", "category": "Image Generation", "pricing_tier": "Freemium",
     "best_for": "Commercially-safe, licensed image generation for business use",
     "tags": "image generation design graphics commercial safe adobe product"},
    {"tool_name": "DALL-E", "category": "Image Generation", "pricing_tier": "Freemium",
     "best_for": "Quick, prompt-accurate images integrated with ChatGPT",
     "tags": "image generation illustration graphics visuals design"},
    {"tool_name": "Ideogram", "category": "Image Generation", "pricing_tier": "Freemium",
     "best_for": "Image generation with accurate embedded text and typography",
     "tags": "image generation logo text typography graphics design"},
    {"tool_name": "Leonardo AI", "category": "Image Generation", "pricing_tier": "Freemium",
     "best_for": "Game assets, concept art, and stylized visual generation",
     "tags": "image generation game art concept art design visuals"},
    {"tool_name": "Higgsfield", "category": "Image Generation", "pricing_tier": "Freemium",
     "best_for": "Dynamic, motion-inspired AI images and short animations",
     "tags": "image generation dynamic motion creative visuals animation"},

    # --- Video Generation ---
    {"tool_name": "Runway", "category": "Video Generation", "pricing_tier": "Paid",
     "best_for": "AI video generation and editing for creative/marketing video",
     "tags": "video generation editing effects creative marketing clip"},
    {"tool_name": "HeyGen", "category": "Video Generation", "pricing_tier": "Freemium",
     "best_for": "AI avatar presenter videos with dubbing and localization",
     "tags": "video avatar dubbing localization marketing explainer presenter"},
    {"tool_name": "Synthesia", "category": "Video Generation", "pricing_tier": "Paid",
     "best_for": "AI avatar training and corporate videos from a script",
     "tags": "video avatar presenter training corporate explainer script"},
    {"tool_name": "Veo", "category": "Video Generation", "pricing_tier": "Freemium",
     "best_for": "High-quality text-to-video generation from Google",
     "tags": "video generation text to video google realistic"},
    {"tool_name": "Pika", "category": "Video Generation", "pricing_tier": "Freemium",
     "best_for": "Short AI-generated video clips from text or images",
     "tags": "video generation short clips text to video"},
    {"tool_name": "Luma AI", "category": "Video Generation", "pricing_tier": "Freemium",
     "best_for": "Realistic AI video generation and 3D capture",
     "tags": "video generation 3d capture realistic visuals"},
    {"tool_name": "Submagic", "category": "Video Generation", "pricing_tier": "Freemium",
     "best_for": "Automatic captions and edits for short-form video platforms",
     "tags": "video captions editing short-form social media clips"},
    {"tool_name": "Pippit", "category": "Video Generation", "pricing_tier": "Freemium",
     "best_for": "Animating static images into talking marketing videos",
     "tags": "video animation marketing social media images talking"},

    # --- Voice & Audio ---
    {"tool_name": "ElevenLabs", "category": "Voice & Audio", "pricing_tier": "Freemium",
     "best_for": "The most realistic AI voice generation and cloning available",
     "tags": "voice audio text to speech narration dubbing podcast cloning"},
    {"tool_name": "Murf AI", "category": "Voice & Audio", "pricing_tier": "Freemium",
     "best_for": "Voiceovers for presentations, training videos, and ads",
     "tags": "voice audio voiceover narration training video ads"},
    {"tool_name": "Descript", "category": "Voice & Audio", "pricing_tier": "Freemium",
     "best_for": "Podcast and video editing by editing a text transcript",
     "tags": "audio video editing podcast transcript voice cloning"},
    {"tool_name": "Podcastle", "category": "Voice & Audio", "pricing_tier": "Freemium",
     "best_for": "AI-powered podcast recording cleanup and enhancement",
     "tags": "audio podcast cleanup recording enhancement voice"},

    # --- Meetings & Productivity ---
    {"tool_name": "Otter.ai", "category": "Meetings & Productivity", "pricing_tier": "Freemium",
     "best_for": "Meeting transcription with automatic summaries and action items",
     "tags": "meeting transcription notes summary action items productivity call"},
    {"tool_name": "Fireflies.ai", "category": "Meetings & Productivity", "pricing_tier": "Freemium",
     "best_for": "Meeting recording, transcription, and CRM integration",
     "tags": "meeting transcription notes crm sales productivity"},
    {"tool_name": "Fathom", "category": "Meetings & Productivity", "pricing_tier": "Freemium",
     "best_for": "The best free meeting recorder with zero setup friction",
     "tags": "meeting recording transcription notes free summary"},
    {"tool_name": "Granola", "category": "Meetings & Productivity", "pricing_tier": "Freemium",
     "best_for": "Meeting notes that blend your own typing with AI transcription",
     "tags": "meeting notes transcription productivity hybrid"},
    {"tool_name": "Notion AI", "category": "Meetings & Productivity", "pricing_tier": "Paid",
     "best_for": "Writing, summarizing, and organizing content inside Notion docs",
     "tags": "productivity notes writing summarizing organization docs workspace"},
    {"tool_name": "Motion", "category": "Meetings & Productivity", "pricing_tier": "Paid",
     "best_for": "AI-automated calendar and task scheduling",
     "tags": "productivity scheduling calendar tasks automation planning"},

    # --- Design ---
    {"tool_name": "Canva", "category": "Design", "pricing_tier": "Freemium",
     "best_for": "The easiest way to turn an idea into graphics, slides, and social assets",
     "tags": "design graphics social media templates marketing visuals easy"},
    {"tool_name": "Figma AI", "category": "Design", "pricing_tier": "Freemium",
     "best_for": "AI-assisted UI/UX design and product prototyping",
     "tags": "design ui ux prototyping product design"},

    # --- Data Analysis & BI ---
    {"tool_name": "Julius AI", "category": "Data Analysis", "pricing_tier": "Freemium",
     "best_for": "Conversational data analysis and visualization from spreadsheets",
     "tags": "data analysis spreadsheet visualization statistics charts query"},
    {"tool_name": "Power BI Copilot", "category": "Data Analysis", "pricing_tier": "Paid",
     "best_for": "AI-assisted report building inside Microsoft Power BI",
     "tags": "data analysis dashboard business intelligence reporting microsoft"},
    {"tool_name": "Akkio", "category": "Data Analysis", "pricing_tier": "Paid",
     "best_for": "No-code predictive modeling for business teams",
     "tags": "data analysis predictive modeling machine learning no-code business"},
    {"tool_name": "Claude (Data Analysis)", "category": "Data Analysis", "pricing_tier": "Freemium",
     "best_for": "Analyzing datasets and building charts/dashboards conversationally",
     "tags": "data analysis csv excel charts dashboard statistics python code"},

    # --- Automation & Agents ---
    {"tool_name": "Zapier", "category": "Automation & Agents", "pricing_tier": "Freemium",
     "best_for": "The widest app library for connecting tools into automated workflows",
     "tags": "automation workflow integration apps trigger no-code agent"},
    {"tool_name": "Make", "category": "Automation & Agents", "pricing_tier": "Freemium",
     "best_for": "Visual, more generous free-tier automation workflows",
     "tags": "automation workflow integration visual builder no-code"},
    {"tool_name": "n8n", "category": "Automation & Agents", "pricing_tier": "Freemium",
     "best_for": "Self-hosted automation for full data control",
     "tags": "automation workflow self-hosted open source data control"},
    {"tool_name": "Craze", "category": "Automation & Agents", "pricing_tier": "Freemium",
     "best_for": "One workspace to chat across models and build no-code agents",
     "tags": "automation agent workspace chat models no-code workflow"},

    # --- Marketing Content ---
    {"tool_name": "Jasper", "category": "Marketing Content", "pricing_tier": "Paid",
     "best_for": "On-brand marketing copy at scale for teams",
     "tags": "marketing copywriting ad copy brand voice content calendar"},
    {"tool_name": "Copy.ai", "category": "Marketing Content", "pricing_tier": "Freemium",
     "best_for": "Short-form marketing copy, social captions, product descriptions",
     "tags": "marketing copywriting social media captions product descriptions"},
    {"tool_name": "AdCreative.ai", "category": "Marketing Content", "pricing_tier": "Paid",
     "best_for": "Generating ad creative variations optimized for conversion",
     "tags": "marketing ads creative variations conversion testing"},

    # --- SEO ---
    {"tool_name": "Semrush", "category": "SEO", "pricing_tier": "Paid",
     "best_for": "The most complete all-in-one SEO platform with AI search tracking",
     "tags": "seo search visibility keywords ranking analytics ai overview"},
    {"tool_name": "Ahrefs", "category": "SEO", "pricing_tier": "Paid",
     "best_for": "Backlink intelligence and competitive SEO analysis",
     "tags": "seo backlinks competitive analysis keywords ranking"},
    {"tool_name": "Surfer SEO", "category": "SEO", "pricing_tier": "Paid",
     "best_for": "On-page content optimization for search ranking",
     "tags": "seo content optimization on-page writing ranking"},

    # --- Customer Support ---
    {"tool_name": "Intercom Fin", "category": "Customer Support", "pricing_tier": "Paid",
     "best_for": "The highest resolution rates for AI-automated customer support",
     "tags": "customer support chatbot resolution tickets automation"},
    {"tool_name": "Freshdesk Freddy AI", "category": "Customer Support", "pricing_tier": "Freemium",
     "best_for": "Best-value AI customer support for growing teams",
     "tags": "customer support chatbot tickets automation helpdesk"},

    # --- Enterprise Knowledge ---
    {"tool_name": "Glean", "category": "Enterprise Knowledge", "pricing_tier": "Paid",
     "best_for": "Enterprise knowledge search across all a company's internal tools",
     "tags": "enterprise search knowledge internal documents company"},

    # --- Document & Utility Tools ---
    {"tool_name": "ChatPDF", "category": "Document Tools", "pricing_tier": "Freemium",
     "best_for": "Asking questions about a PDF instead of reading it page by page",
     "tags": "document pdf questions reading summarization"},
    {"tool_name": "TinyWow", "category": "Document Tools", "pricing_tier": "Free",
     "best_for": "Dozens of everyday file conversion and editing utilities",
     "tags": "document utility conversion file editing free"},

    # --- Music Generation ---
    {"tool_name": "Suno", "category": "Music Generation", "pricing_tier": "Freemium",
     "best_for": "Generating full songs with vocals from a text prompt",
     "tags": "music song generation vocals audio creative"},
    {"tool_name": "Udio", "category": "Music Generation", "pricing_tier": "Freemium",
     "best_for": "AI music generation with genre and style control",
     "tags": "music song generation genre style audio creative"},

    # --- Creative & General Writing ---
    {"tool_name": "Sudowrite", "category": "Creative Writing", "pricing_tier": "Paid",
     "best_for": "Fiction writing, story brainstorming, and creative prose",
     "tags": "fiction creative writing story novel brainstorming prose"},
    {"tool_name": "Grammarly", "category": "Writing & Editing", "pricing_tier": "Freemium",
     "best_for": "Grammar, tone, and clarity editing on existing text",
     "tags": "grammar editing proofreading tone clarity writing assistant"},
    {"tool_name": "Writesonic", "category": "Writing & Editing", "pricing_tier": "Freemium",
     "best_for": "General AI writing for blogs, ads, and long-form content",
     "tags": "writing blog content long-form ads general"},
    {"tool_name": "QuillBot", "category": "Writing & Editing", "pricing_tier": "Freemium",
     "best_for": "Paraphrasing, summarizing, and grammar checking for students and writers",
     "tags": "writing paraphrase summarize grammar student editing"},
    {"tool_name": "Wordtune", "category": "Writing & Editing", "pricing_tier": "Freemium",
     "best_for": "Rewriting sentences for clarity and tone on the fly",
     "tags": "writing rewrite tone clarity editing sentence"},

    # --- More Coding ---
    {"tool_name": "Tabnine", "category": "Coding & Development", "pricing_tier": "Freemium",
     "best_for": "Privacy-focused AI code completion with on-premise deployment options",
     "tags": "coding autocomplete privacy on-premise enterprise developer"},

    # --- More Presentations ---
    {"tool_name": "Plus AI", "category": "Presentations", "pricing_tier": "Freemium",
     "best_for": "AI slide generation directly inside Google Slides and PowerPoint",
     "tags": "presentation slides google powerpoint plugin generate"},
    {"tool_name": "SlidesAI", "category": "Presentations", "pricing_tier": "Freemium",
     "best_for": "Turning text into Google Slides presentations automatically",
     "tags": "presentation slides google automatic text to slides"},

    # --- More Image ---
    {"tool_name": "Stable Diffusion", "category": "Image Generation", "pricing_tier": "Free",
     "best_for": "Open-source, self-hostable image generation with full control",
     "tags": "image generation open source self-hosted customizable art"},

    # --- More Video ---
    {"tool_name": "Kapwing", "category": "Video Generation", "pricing_tier": "Freemium",
     "best_for": "Browser-based collaborative video editing with AI tools built in",
     "tags": "video editing collaborative browser social media clips"},
    {"tool_name": "InVideo", "category": "Video Generation", "pricing_tier": "Freemium",
     "best_for": "Turning scripts or blog posts into full marketing videos",
     "tags": "video generation marketing script blog templates"},

    # --- More Voice ---
    {"tool_name": "Play.ht", "category": "Voice & Audio", "pricing_tier": "Freemium",
     "best_for": "Ultra-realistic text-to-speech for apps and content",
     "tags": "voice audio text to speech api realistic narration"},
    {"tool_name": "Krisp", "category": "Voice & Audio", "pricing_tier": "Freemium",
     "best_for": "Real-time AI noise cancellation for calls and recordings",
     "tags": "audio noise cancellation calls meetings clarity"},

    # --- More Meetings ---
    {"tool_name": "Supernormal", "category": "Meetings & Productivity", "pricing_tier": "Freemium",
     "best_for": "Automatic, well-formatted meeting notes with minimal setup",
     "tags": "meeting notes transcription productivity automatic"},

    # --- More Design ---
    {"tool_name": "Uizard", "category": "Design", "pricing_tier": "Freemium",
     "best_for": "Turning sketches or text prompts into UI mockups fast",
     "tags": "design ui mockup prototyping sketch app screens"},
    {"tool_name": "Looka", "category": "Design", "pricing_tier": "Paid",
     "best_for": "AI logo and brand identity generation for small businesses",
     "tags": "design logo branding identity small business"},

    # --- More Data Analysis ---
    {"tool_name": "Tableau AI", "category": "Data Analysis", "pricing_tier": "Paid",
     "best_for": "AI-assisted insight generation inside enterprise Tableau dashboards",
     "tags": "data analysis dashboard business intelligence enterprise visualization"},
    {"tool_name": "DataRobot", "category": "Data Analysis", "pricing_tier": "Paid",
     "best_for": "Enterprise-grade automated machine learning pipelines",
     "tags": "data analysis machine learning automated enterprise predictive"},

    # --- More Automation ---
    {"tool_name": "Relay.app", "category": "Automation & Agents", "pricing_tier": "Freemium",
     "best_for": "Human-in-the-loop automation workflows with AI steps",
     "tags": "automation workflow human review agent no-code"},
    {"tool_name": "Gumloop", "category": "Automation & Agents", "pricing_tier": "Freemium",
     "best_for": "Visual AI workflow builder for non-technical automation",
     "tags": "automation workflow visual builder no-code agent"},

    # --- More Marketing ---
    {"tool_name": "Anyword", "category": "Marketing Content", "pricing_tier": "Paid",
     "best_for": "Performance-predicting marketing copy generation",
     "tags": "marketing copywriting performance prediction ads content"},

    # --- More SEO ---
    {"tool_name": "Clearscope", "category": "SEO", "pricing_tier": "Paid",
     "best_for": "Content optimization graded against top-ranking pages",
     "tags": "seo content optimization grading ranking keywords"},

    # --- More Customer Support ---
    {"tool_name": "Zendesk AI", "category": "Customer Support", "pricing_tier": "Paid",
     "best_for": "AI-automated ticket routing and resolution at enterprise scale",
     "tags": "customer support chatbot tickets automation enterprise helpdesk"},
    {"tool_name": "Ada", "category": "Customer Support", "pricing_tier": "Paid",
     "best_for": "No-code AI customer service agents for high ticket volume",
     "tags": "customer support chatbot automation no-code agent"},

    # --- More Enterprise Knowledge ---
    {"tool_name": "Guru", "category": "Enterprise Knowledge", "pricing_tier": "Freemium",
     "best_for": "Verified internal knowledge base surfaced where teams already work",
     "tags": "enterprise knowledge internal documents verified wiki"},

    # --- More Document Tools ---
    {"tool_name": "PDF.ai", "category": "Document Tools", "pricing_tier": "Freemium",
     "best_for": "Chatting with PDFs to extract answers and summaries",
     "tags": "document pdf chat questions summarization extraction"},
    {"tool_name": "Adobe Acrobat AI Assistant", "category": "Document Tools", "pricing_tier": "Freemium",
     "best_for": "AI summarization and Q&A built into the standard PDF workflow",
     "tags": "document pdf summarization questions adobe standard"},

    # --- More Music ---
    {"tool_name": "AIVA", "category": "Music Generation", "pricing_tier": "Freemium",
     "best_for": "Composing original instrumental scores for film and games",
     "tags": "music composition instrumental score film game"},

    # --- More Creative Writing ---
    {"tool_name": "NovelAI", "category": "Creative Writing", "pricing_tier": "Paid",
     "best_for": "AI-assisted long-form fiction and roleplay writing",
     "tags": "fiction creative writing novel roleplay story"},

    # --- New Category: Legal & Compliance ---
    {"tool_name": "Harvey AI", "category": "Legal & Compliance", "pricing_tier": "Paid",
     "best_for": "AI-assisted legal research, contract review, and drafting for law firms",
     "tags": "legal contract review research drafting compliance law"},

    # --- New Category: Email & Sales ---
    {"tool_name": "Superhuman AI", "category": "Email & Sales", "pricing_tier": "Paid",
     "best_for": "AI-triaged, fast email management for busy professionals",
     "tags": "email inbox triage productivity sales fast"},

    # --- New Category: Translation ---
    {"tool_name": "DeepL", "category": "Translation", "pricing_tier": "Freemium",
     "best_for": "The most accurate AI translation across major languages",
     "tags": "translation language accurate documents text"},

    # --- New Category: Notes & Knowledge Management ---
    {"tool_name": "Mem", "category": "Notes & Knowledge Management", "pricing_tier": "Freemium",
     "best_for": "AI-organized personal notes that self-sort without folders",
     "tags": "notes knowledge management organization personal search"},

    # --- New Category: Avatar & Digital Human ---
    {"tool_name": "D-ID", "category": "Avatar & Digital Human", "pricing_tier": "Freemium",
     "best_for": "Turning a photo into a talking digital avatar video",
     "tags": "avatar digital human video talking photo animation"},

    # --- New Category: Website Building ---
    {"tool_name": "Framer AI", "category": "Website Building", "pricing_tier": "Freemium",
     "best_for": "AI-generated, designer-quality websites from a text prompt",
     "tags": "website builder design no-code landing page prompt"},
    {"tool_name": "Glide", "category": "Website Building", "pricing_tier": "Freemium",
     "best_for": "Turning a spreadsheet into a working app or internal tool",
     "tags": "app builder no-code spreadsheet internal tool website"},

    # --- New Category: 3D & Spatial Design ---
    {"tool_name": "Spline AI", "category": "3D & Spatial Design", "pricing_tier": "Freemium",
     "best_for": "AI-assisted 3D design and interactive web graphics",
     "tags": "3d design spatial interactive web graphics animation"},
]

URL_MAP = {
    "ChatGPT": "https://chatgpt.com", "Claude": "https://claude.ai", "Gemini": "https://gemini.google.com",
    "Grok": "https://grok.com", "Microsoft Copilot": "https://copilot.microsoft.com",
    "Cursor": "https://cursor.com", "Claude Code": "https://claude.com/product/claude-code",
    "GitHub Copilot": "https://github.com/features/copilot", "Replit AI": "https://replit.com",
    "Aider": "https://aider.chat", "Windsurf": "https://windsurf.com", "Lovable": "https://lovable.dev",
    "Bolt.new": "https://bolt.new", "v0": "https://v0.dev",
    "Perplexity": "https://www.perplexity.ai", "NotebookLM": "https://notebooklm.google",
    "Consensus": "https://consensus.app", "Elicit": "https://elicit.com", "You.com": "https://you.com",
    "Gamma": "https://gamma.app", "Beautiful.ai": "https://www.beautiful.ai", "Tome": "https://tome.app",
    "Decktopus": "https://www.decktopus.com",
    "Midjourney": "https://www.midjourney.com", "Adobe Firefly": "https://firefly.adobe.com",
    "DALL-E": "https://openai.com/dall-e-3", "Ideogram": "https://ideogram.ai",
    "Leonardo AI": "https://leonardo.ai", "Higgsfield": "https://higgsfield.ai",
    "Runway": "https://runwayml.com", "HeyGen": "https://www.heygen.com", "Synthesia": "https://www.synthesia.io",
    "Veo": "https://deepmind.google/technologies/veo/", "Pika": "https://pika.art",
    "Luma AI": "https://lumalabs.ai", "Submagic": "https://www.submagic.co", "Pippit": "https://www.pippit.ai",
    "ElevenLabs": "https://elevenlabs.io", "Murf AI": "https://murf.ai", "Descript": "https://www.descript.com",
    "Podcastle": "https://podcastle.ai",
    "Otter.ai": "https://otter.ai", "Fireflies.ai": "https://fireflies.ai", "Fathom": "https://fathom.video",
    "Granola": "https://www.granola.ai", "Notion AI": "https://www.notion.com/product/ai",
    "Motion": "https://www.usemotion.com",
    "Canva": "https://www.canva.com", "Figma AI": "https://www.figma.com/ai",
    "Julius AI": "https://julius.ai", "Power BI Copilot": "https://powerbi.microsoft.com",
    "Akkio": "https://www.akkio.com", "Claude (Data Analysis)": "https://claude.ai",
    "Zapier": "https://zapier.com", "Make": "https://www.make.com", "n8n": "https://n8n.io",
    "Craze": "https://www.crazehq.com",
    "Jasper": "https://www.jasper.ai", "Copy.ai": "https://www.copy.ai", "AdCreative.ai": "https://www.adcreative.ai",
    "Semrush": "https://www.semrush.com", "Ahrefs": "https://ahrefs.com", "Surfer SEO": "https://surferseo.com",
    "Intercom Fin": "https://www.intercom.com/fin", "Freshdesk Freddy AI": "https://www.freshworks.com/freshdesk",
    "Glean": "https://www.glean.com",
    "ChatPDF": "https://www.chatpdf.com", "TinyWow": "https://tinywow.com",
    "Suno": "https://suno.com", "Udio": "https://www.udio.com",
    "Sudowrite": "https://www.sudowrite.com", "Grammarly": "https://www.grammarly.com",
    "Writesonic": "https://writesonic.com", "QuillBot": "https://quillbot.com", "Wordtune": "https://www.wordtune.com",
    "Tabnine": "https://www.tabnine.com", "Plus AI": "https://plusai.com", "SlidesAI": "https://www.slidesai.io",
    "Stable Diffusion": "https://stability.ai", "Kapwing": "https://www.kapwing.com", "InVideo": "https://invideo.io",
    "Play.ht": "https://play.ht", "Krisp": "https://krisp.ai", "Supernormal": "https://supernormal.com",
    "Uizard": "https://uizard.io", "Looka": "https://looka.com",
    "Tableau AI": "https://www.tableau.com", "DataRobot": "https://www.datarobot.com",
    "Relay.app": "https://relay.app", "Gumloop": "https://www.gumloop.com",
    "Anyword": "https://www.anyword.com", "Clearscope": "https://www.clearscope.io",
    "Zendesk AI": "https://www.zendesk.com/service/ai", "Ada": "https://www.ada.cx",
    "Guru": "https://www.getguru.com",
    "PDF.ai": "https://pdf.ai", "Adobe Acrobat AI Assistant": "https://www.adobe.com/acrobat/generative-ai-pdf.html",
    "AIVA": "https://www.aiva.ai", "NovelAI": "https://novelai.net",
    "Harvey AI": "https://www.harvey.ai", "Superhuman AI": "https://superhuman.com",
    "DeepL": "https://www.deepl.com", "Mem": "https://mem.ai", "D-ID": "https://www.d-id.com",
    "Framer AI": "https://www.framer.com/ai", "Glide": "https://www.glideapps.com", "Spline AI": "https://spline.design",
}

df = pd.DataFrame(tools)
df["official_url"] = df["tool_name"].map(URL_MAP)
missing = df[df["official_url"].isna()]["tool_name"].tolist()
if missing:
    print(f"WARNING - no URL mapped for: {missing}")
df.to_csv("ai_tools_catalog.csv", index=False)
print(f"Catalog built: {len(df)} tools across {df['category'].nunique()} categories")
print(df["category"].value_counts())
