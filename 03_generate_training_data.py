"""
STEP 3: Labeled training data - substantially expanded and targeted.
Previous version had 8-16 examples/category (63.2% accuracy) with several
categories at 0% recall due to tiny test sets (2 examples) as much as model
weakness. This version:
1. Roughly doubles/triples volume per category (target 20-35 each)
2. Adds extra volume specifically to the weakest categories from the last
   confusion matrix: Data Analysis, Music Generation, Search & Research,
   Writing & Editing, General AI Assistant, Coding & Development
3. Uses more category-distinctive vocabulary to reduce overlap between
   commonly-confused neighbors (e.g. Data Analysis vs Coding & Development)
"""
import pandas as pd
import random

random.seed(7)

TRAINING_EXAMPLES = {
    "General AI Assistant": [
        "help me think through a hard decision", "answer a general knowledge question",
        "I need a versatile assistant for anything", "have a conversation to brainstorm ideas",
        "ask a broad question about many topics", "I need one tool that does a bit of everything",
        "get help reasoning through a problem", "chat with an AI about my day",
        "I want a general purpose chatbot", "ask a wide-ranging question about the world",
        "have a back and forth conversation to work through my thinking", "get a second opinion on my reasoning",
        "I need a smart assistant for random daily questions", "talk through a tricky situation with an AI",
        "explain a complex topic to me simply", "help me weigh the pros and cons of a choice",
        "I want to bounce ideas off an AI", "ask a philosophical question and discuss it",
        "get advice on a personal situation", "have an open ended conversation about anything",
        "help me understand something I'm confused about", "talk through my options before deciding",
        "I need a thinking partner for a big decision", "chat casually about a random topic",
        "ask general trivia or knowledge questions"
    ],
    "Coding & Development": [
        "help me fix a bug in my code", "write a function to process this data",
        "I need help debugging this script", "review my code for errors",
        "help me refactor this messy function", "complete this half-written program",
        "generate boilerplate code for an API", "help me write unit tests",
        "I need to build a small app quickly", "autocomplete this code as I type",
        "write a python script to parse this file", "help me understand this error message",
        "generate a sql query for this database", "pair program with me on this feature",
        "help me optimize this slow function", "write code to automate this task",
        "implement a new feature in this codebase", "help me set up a git repository",
        "write a javascript function for this website", "fix this compiler error",
        "review my pull request for bugs", "help me write a class in object oriented code",
        "translate this code from one language to another", "build a REST API endpoint",
        "write a recursive algorithm for this problem", "help me trace through this stack trace",
        "generate code comments and documentation", "build a command line tool in python",
        "help me set up unit testing for my project", "write code to connect to a database",
        "fix a syntax error in my program"
    ],
    "Search & Research": [
        "find recent studies on this topic", "I need sources for a research paper",
        "look up what scientists say about this", "search for the latest news on this topic",
        "I need citations for my thesis", "find evidence supporting this claim",
        "research the current state of this field", "find credible sources for a report",
        "search the web for current information", "look up facts and verify with sources",
        "find academic papers on this subject", "get real-time search results with citations",
        "look up who won an award or event", "search for statistics on a topic with sources",
        "find out what happened in the news today", "research a company's background with sources",
        "look up the history of a topic with citations", "verify a fact by searching multiple sources",
        "find peer reviewed research on a question", "search for expert opinions on a topic",
        "look up current prices or rankings with sources", "research a topic before writing about it",
        "find out the latest developments in a field", "search and summarize multiple articles on a topic",
        "fact check a claim using web sources"
    ],
    "Presentations": [
        "turn my notes into a slide deck", "I need slides for an investor pitch",
        "make a presentation from this outline", "build a deck for tomorrow's meeting",
        "convert this document into slides quickly", "I need a pitch deck for my startup",
        "help me design a professional slideshow", "create slides summarizing this report",
        "generate a slide deck from this text", "make a business presentation quickly",
        "build slides for a client meeting", "turn a google doc into slides",
        "design slides for a conference talk", "create a slideshow from bullet points",
        "make a quarterly review presentation", "build a sales pitch deck",
        "generate slides with charts and data", "create a training presentation deck",
        "design a slide deck for a webinar"
    ],
    "Image Generation": [
        "create a picture for my blog post", "generate an illustration for my book cover",
        "I need product photos for my online store", "make artwork for a poster",
        "design a logo for my company", "generate a fantasy landscape image",
        "create social media graphics", "generate images for my presentation",
        "make an ai generated picture of a character", "create a stylized digital painting",
        "generate a photorealistic image from a description", "make icon graphics for my app",
        "create concept art for a game character", "generate a picture of an imaginary scene",
        "design album cover artwork", "create an illustration in a specific art style",
        "generate a picture from a text description", "make a digital portrait of a character",
        "create abstract art for a wall print", "generate a texture or pattern image"
    ],
    "Video Generation": [
        "create a short promotional video", "I need an explainer video for my product",
        "make a training video with a presenter", "generate a video from this script",
        "I want an avatar to present my content", "create marketing video clips",
        "turn this text into a video", "add captions to my short-form video",
        "generate a video ad for social media", "make a text to video clip",
        "produce a video from a blog post automatically", "create a video with an ai spokesperson",
        "generate a cinematic video clip from a prompt", "animate a scene into a short film",
        "create a product demo video", "generate a video with realistic motion",
        "make a video walkthrough of my app", "produce a highlight reel automatically",
        "create a talking head video from a script"
    ],
    "Voice & Audio": [
        "generate a voiceover for my video", "I need a realistic AI narrator voice",
        "turn this text into spoken audio", "clone a voice for my podcast intro",
        "remove background noise from my call recording", "generate narration for an audiobook",
        "I need voice for a training module", "clean up my podcast recording",
        "convert text to natural sounding speech", "create a synthetic voice for my app",
        "improve audio quality on my recording", "generate a voiceover in multiple languages",
        "create a custom AI voice for my brand", "record and enhance a podcast episode",
        "generate an audiobook narration from a manuscript", "make an announcement voice for my app",
        "remove echo and static from an audio file", "generate a voice for an IVR phone system"
    ],
    "Meetings & Productivity": [
        "transcribe my meeting recording", "summarize this call into action items",
        "I need notes from today's meeting", "organize my notes automatically",
        "get a summary of what was discussed on the call", "log meeting minutes automatically",
        "help me schedule my calendar automatically", "extract key decisions from this meeting",
        "record and transcribe my zoom call", "automatically block time on my calendar",
        "capture follow-up tasks from a meeting", "turn a meeting recording into notes",
        "summarize a long conference call", "generate meeting minutes automatically",
        "track action items across multiple meetings", "find time slots that work for everyone",
        "get a written record of what was said in a call", "automatically organize my daily schedule",
        "record a video call and generate a transcript", "get action items emailed after every meeting"
    ],
    "Design": [
        "help me make a flyer for an event", "create a UI mockup for my app",
        "design a banner ad", "I need a quick graphic for instagram",
        "prototype a mobile app screen", "design a logo for my small business",
        "turn my sketch into a UI design", "create a marketing graphic quickly",
        "design a business card", "mock up a new app screen from a description",
        "create branding assets for my company", "design a poster for an event",
        "design a wireframe for a new feature", "create a style guide for my brand",
        "design packaging artwork for a product", "build a clickable prototype of an app screen",
        "design an email newsletter template", "create a consistent visual identity for my brand"
    ],
    "Data Analysis": [
        "analyze this spreadsheet and find trends", "help me build a chart from this csv",
        "summarize the key patterns in this dataset", "build a dashboard from this data",
        "run statistics on this data file", "clean and analyze this messy dataset",
        "predict future values from this data", "visualize this sales data",
        "find correlations between these columns", "generate a report from this dataset",
        "build a predictive model from this data", "create a business intelligence dashboard",
        "calculate summary statistics for this spreadsheet", "find outliers in this dataset",
        "build a pivot table from this data", "analyze survey results and summarize findings",
        "create a forecast based on historical numbers", "segment customers based on this data",
        "run a regression analysis on this dataset", "clean messy excel data and remove duplicates",
        "build a chart comparing performance across regions", "calculate year over year growth from this data",
        "analyze website traffic data for trends", "find the average and median of this dataset",
        "build an interactive chart from spreadsheet data", "identify which factors most affect this outcome",
        "aggregate this data by category and summarize"
    ],
    "Automation & Agents": [
        "connect two apps to automate a task", "I want an agent to handle repetitive work",
        "automate my workflow between tools", "set up a trigger when a form is submitted",
        "build a no-code automation", "let an AI agent run this task on autopilot",
        "sync data automatically between two systems", "automate sending emails when something happens",
        "build a workflow that runs without me", "connect my apps so data flows automatically",
        "set up an automated pipeline between tools", "create an AI agent to handle a repetitive process",
        "automatically move data from one app to another", "trigger an action when a new row is added",
        "set up a zap between my apps", "build a multi-step automated workflow",
        "automate a repetitive manual process at work", "create a bot that performs a task on a schedule"
    ],
    "Marketing Content": [
        "write ad copy for my campaign", "generate product descriptions for my store",
        "write social media captions", "create email marketing copy",
        "write a catchy tagline for my brand", "generate ad variations for testing",
        "write a product launch announcement", "predict which ad copy will perform best",
        "write a marketing email sequence", "generate copy for a landing page",
        "write a compelling product description", "create ad copy optimized for conversions",
        "write copy for a facebook ad campaign", "generate multiple headline options for an ad",
        "write a promotional email for a sale", "create copy for a product launch campaign"
    ],
    "SEO": [
        "check my website's keyword rankings", "find backlink opportunities for my site",
        "optimize this article for search ranking", "research keywords for a blog post",
        "see how my site compares to competitors", "grade my content against top-ranking pages",
        "track my visibility in AI search results", "audit my site's SEO performance",
        "find high-value keywords for my niche", "improve my page's search ranking",
        "analyze competitor backlink profiles", "optimize on-page content for search engines",
        "find low competition keywords to target", "audit my website for technical SEO issues",
        "track how my rankings change over time", "research what keywords competitors rank for",
        "improve my site's page speed for search ranking", "find content gaps compared to competitors"
    ],
    "Customer Support": [
        "automate responses to customer tickets", "build a chatbot for customer service",
        "reduce our support ticket resolution time", "handle refund requests automatically",
        "route support tickets to the right team", "deflect common customer questions automatically",
        "set up 24/7 automated customer support", "resolve tickets without a human agent",
        "build an AI agent to answer support questions", "automate our helpdesk ticket triage",
        "reduce customer wait times with automation", "create a self-service support bot",
        "automatically categorize incoming support tickets", "build an FAQ chatbot for our website",
        "reduce the volume of repetitive support tickets", "set up automated ticket escalation rules"
    ],
    "Document Tools": [
        "ask questions about this PDF", "summarize this long document quickly",
        "extract data from a scanned form", "convert this file to another format",
        "get answers from a contract without reading it fully", "summarize this report for me",
        "chat with a PDF to find specific information", "extract key clauses from this document",
        "convert a pdf into an editable file", "extract tables from a pdf document",
        "get a quick summary of a lengthy pdf", "search inside a pdf for specific information",
        "merge multiple pdf files into one", "extract text from a scanned image document",
        "convert a word document to pdf", "find a specific clause in a long contract",
        "compress a large pdf file", "split a pdf into separate pages",
        "convert a scanned receipt into text data", "extract a table of numbers from a pdf report"
    ],
    "Writing & Editing": [
        "check my writing for grammar mistakes", "paraphrase this paragraph",
        "rewrite this sentence to sound clearer", "proofread my essay",
        "improve the tone of this email", "summarize this article in my own words",
        "fix awkward phrasing in my writing", "polish this document before I send it",
        "correct spelling and grammar errors", "rewrite this to sound more professional",
        "improve clarity of this paragraph", "check this text for tone and readability",
        "shorten this paragraph without losing meaning", "make this writing sound more formal",
        "fix run-on sentences in my essay", "check my resume for grammar errors",
        "simplify this technical paragraph for a general audience", "rewrite this to remove passive voice",
        "improve the flow between these paragraphs", "check this cover letter for mistakes",
        "make my writing more concise", "fix punctuation errors in this text",
        "improve word choice in this paragraph", "check consistency of tense throughout this document"
    ],
    "Music Generation": [
        "generate a song for my video background", "create background music for my podcast",
        "compose a melody for my project", "generate royalty free music",
        "make a song with vocals from lyrics", "compose an instrumental score for my film",
        "create a jingle for my brand", "generate music in a specific genre",
        "make an original soundtrack for a game", "compose a theme song from a description",
        "generate a lo-fi beat for studying", "create a piece of orchestral music",
        "compose background music for a youtube video", "generate an upbeat song for an ad",
        "create a custom ringtone melody", "compose music that matches a certain mood",
        "generate a full song from a text prompt", "create a short musical sting for an intro"
    ],
    "Enterprise Knowledge": [
        "search across all our internal company documents", "find information buried in our company wiki",
        "search knowledge across all our internal tools at once", "surface verified answers from our internal docs",
        "search our company's internal knowledge base", "find the right internal document quickly",
        "unify search across our company's scattered tools", "get verified answers from internal sources",
        "search across slack, drive, and confluence at once", "find who owns a project across our company systems",
        "surface the most up to date internal policy document", "search internal engineering documentation",
        "find an internal document nobody can locate", "search across every internal tool from one place"
    ],
    "Creative Writing": [
        "help me write a short story", "brainstorm plot ideas for my novel",
        "write a fictional dialogue between characters", "continue this story I started",
        "help me write compelling fiction", "write a fantasy story opening",
        "develop a character for my novel", "write a poem about a theme",
        "help me plot out a screenplay", "brainstorm a fictional world for my book",
        "write a short piece of flash fiction", "help me write a plot twist for my story",
        "develop the backstory for a fictional character", "write a scene with tension between characters",
        "brainstorm names for characters in my novel", "write an opening line that hooks readers"
    ],
    "Website Building": [
        "build a website from a text description", "turn my spreadsheet into a working app",
        "create a landing page quickly without coding", "design a website without a developer",
        "build an internal tool from my data", "make a portfolio website fast",
        "generate a website from a prompt", "build a no-code web app quickly",
        "turn a design into a working website", "create a simple app without writing code",
        "build a small business website quickly", "create a signup form connected to a database",
        "build an internal dashboard from a spreadsheet", "launch a simple e-commerce site quickly",
        "publish a website without any coding knowledge", "spin up a working web app from a prompt",
        "build a booking page for my business", "create a website for my small business in minutes"
    ],
    "Legal & Compliance": [
        "review this contract for risky clauses", "help with legal research on this case",
        "draft a standard legal agreement", "check this document for compliance issues",
        "review a legal contract for problem terms", "research case law for this legal question",
        "draft a non-disclosure agreement", "flag risky clauses in a legal document",
        "summarize the key terms of this contract", "check if this document meets regulatory requirements",
        "draft a standard employment agreement", "review terms of service for legal risk",
        "identify liability clauses in this agreement", "draft a vendor services contract"
    ],
    "Email & Sales": [
        "help me manage my overflowing inbox", "triage my emails by priority",
        "draft a fast reply to this email thread", "organize my inbox automatically for sales follow-ups",
        "prioritize which emails need a response first", "speed up how fast I reply to emails",
        "manage my sales inbox efficiently", "triage and sort my daily emails",
        "draft a follow-up email to a sales lead", "clean up and organize a cluttered inbox",
        "flag high priority emails automatically", "draft a quick reply to a client email",
        "summarize a long email thread quickly", "automatically sort emails into folders"
    ],
    "Translation": [
        "translate this document into another language", "get an accurate translation of this text",
        "translate this email to another language", "convert this webpage into a different language",
        "translate this contract into another language", "get a precise translation of this paragraph",
        "translate my website content into another language", "accurately translate this text between languages",
        "translate a menu into another language", "convert subtitles into a different language",
        "translate technical documentation accurately", "get a certified quality translation of a document",
        "translate a product manual into multiple languages", "convert a customer message into my language"
    ],
    "Notes & Knowledge Management": [
        "organize my personal notes automatically", "find an old note without searching folders",
        "self-organize my scattered notes", "search across everything I've written down",
        "automatically tag and organize my notes", "find a note I wrote months ago",
        "keep my personal knowledge organized without folders", "search my personal notes instantly",
        "connect related notes automatically", "build a personal knowledge base from my notes",
        "link related ideas across my notes automatically", "capture a quick thought and file it automatically",
        "turn scattered notes into an organized wiki", "find every note I've written about a topic",
        "keep a running journal that organizes itself", "surface old notes relevant to what I'm working on now"
    ],
    "Avatar & Digital Human": [
        "turn my photo into a talking avatar video", "create a digital human to present my content",
        "animate a photo to talk and present", "make an AI avatar spokesperson video",
        "turn a still photo into a talking video", "create a lifelike digital presenter",
        "animate my headshot to speak a script", "generate a talking avatar from an image",
        "create a virtual presenter for training content", "make a photo lip-sync to an audio track",
        "build a digital twin avatar of myself", "create an AI presenter that speaks my script",
        "generate a virtual human to host a video", "make a static picture speak with lip sync",
        "create a realistic digital avatar for marketing", "animate a headshot into a presenting character"
    ],
    "3D & Spatial Design": [
        "design an interactive 3D scene for my website", "create 3D graphics for a web project",
        "build an interactive 3D model", "design spatial graphics for the web",
        "create an interactive 3D product viewer", "build a 3D scene for my portfolio site",
        "design an interactive web animation in 3D", "create a 3D visualization for the browser",
        "build a rotating 3D product display", "design an immersive 3D web experience",
        "create a 3D model viewer for an e-commerce site", "build an interactive spatial design for the web",
        "design a 3D configurator for a product", "create a webGL scene for my portfolio"
    ],

}

rows = []
for category, examples in TRAINING_EXAMPLES.items():
    for text in examples:
        rows.append({"task_text": text, "category": category})

df = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("task_training_data.csv", index=False)
print(f"Training data built: {len(df)} labeled examples across {df['category'].nunique()} categories")
counts = df["category"].value_counts()
print(f"Min examples per category: {counts.min()}  |  Max: {counts.max()}  |  Mean: {counts.mean():.1f}")
