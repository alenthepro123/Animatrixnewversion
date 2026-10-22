import base64
import html
import os

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "Red_Geass.png")
BACKGROUND_PATH = os.path.join(BASE_DIR, "1319754.jpeg")
ANIME_METADATA_PATH = os.path.join(BASE_DIR, "anime-dataset-2023.csv")
MODEL_PATH = os.path.join(BASE_DIR, "anime_model_full (1).pth")

# --- Page configuration ---
st.set_page_config(
    page_title="ANIMATRIX",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🌌",
    layout="wide",
)

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANIME_TITLES = [
    'Akame ga Kill!', 'Angel Beats!', 'Ano Hi Mita Hana no Namae wo Bokutachi wa Mada Shiranai',
    'Another', 'Ansatsu Kyoushitsu', 'Ao no Exorcist', 'Bakemonogatari', 'BANANA FISH', 'Black Clover',
    'Bleach', 'Boku dake ga Inai Machi', 'Boku no Hero Academia', 'Bungou Stray Dogs', 'Chainsaw Man',
    'Charlotte', 'Chuunibyou demo Koi ga Shitai', 'CLANNAD', 'Code Geass_ Hangyaku no Lelouch',
    'Cowboy Bebop', 'Darling in the Franxx', 'Death Note', 'Death Parade', 'Devilman Crybaby',
    'Dororo', 'Dr. Stone', 'Dungeon ni Deai wo Motomeru no wa Machigatteiru Darou ka', 'Durarara!!',
    'Enen no Shouboutai', 'Fairy Tail', 'Fate Zero', 'Fruits Basket_ 1st Season', 'Fumetsu no Anata e',
    'Go-toubun no Hanayome', 'Goblin Slayer', 'Hagane no Renkinjutsushi_ Fullmetal Alchemist',
    'Haikyuu!!', 'Hataraku Maou-sama!', 'High School DxD', 'Horimiya', 'Howl no Ugoku Shiro',
    'Hunter x Hunter', 'Hyouka', 'JoJo', 'Jujutsu Kaisen', 'Kaguya-sama wa Kokurasetai_ Tensaitachi no Renai Zunousen',
    'Kakegurui', 'Kami no Tou_ Tower of God', 'Kanojo, Okarishimasu', 'Kill la Kill', 'Kimetsu no Yaiba',
    'Kimi no Na wa', 'Kimi no Suizou wo Tabetai', 'Kiseijuu_ Sei no Kakuritsu', 'Kobayashi-san Chi no Maidragon',
    'Koe no Katachi', 'Komi-san wa, Komyushou desu', 'Kono Subarashii Sekai ni Shukufuku wo!', 'Made in Abyss',
    'Mahou Shoujo Madoka Magica', 'Mirai Nikki', 'Mob Psycho 100', 'Monster', 'Mushoku Tensei_ Isekai Ittara Honki Dasu',
    'Nanatsu no Taizai', 'Naruto', 'No Game No Life', 'Noragami', 'One Piece', 'One Punch Man', 'Overlord',
    'Owari no Seraph', 'Psycho-Pass', 'Re_Zero kara Hajimeru Isekai Seikatsu', 'Saiki Kusuo no Psi-nan',
    'Seishun Buta Yarou wa Bunny Girl Senpai no Yume wo Minai', 'Sen to Chihiro no Kamikakushi', 'Shigatsu wa Kimi no Uso',
    'Shin Seiki Evangelion', 'Shingeki no Kyojin', 'Shokugeki no Souma', 'Sono Bisque Doll wa Koi wo Suru',
    'Soul Eater', 'Sousou no Frieren', 'Spy x Family', 'Steins Gate', 'Sword Art Online',
    'Tate no Yuusha no Nariagari', 'Tengen Toppa Gurren Lagann', 'Tenki no Ko', 'Tensei Shitara Slime Datta Ken',
    'The God of High School', 'Tokyo Ghoul', 'Tokyo Revengers', 'Toradora!', 'Vinland Saga', 'Violet Evergarden',
    'Wotaku ni Koi wa Muzukashii', 'Yahari Ore no Seishun Love Come wa Machigatteiru', 'Yakusoku no Neverland',
    'Youkoso Jitsuryoku Shijou Shugi no Kyoushitsu e'
]


# --- Helpers ---
def file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def clean_title(title):
    """Folder names used '_' where the real title has ':'. Show the readable version."""
    return title.replace("_ ", ": ").replace("_", ":")


def html_block(*lines):
    """Join HTML lines with no indentation or blank lines so Streamlit's markdown never treats it as code."""
    return "".join(line.strip() for line in lines)


# --- Styling ---
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Audiowide&family=Exo+2:wght@400;600;800&display=swap');

@property --angle { syntax: '<angle>'; initial-value: 0deg; inherits: false; }
@property --num   { syntax: '<integer>'; initial-value: 0; inherits: true; }

:root {
    --void: #0e0a2c;
    --panel: rgba(26, 18, 70, 0.62);
    --aurora: #5ce1e6;
    --violet: #8f7bff;
    --sakura: #ff6fa8;
    --starlight: #f2efff;
    --dim: #b9b2e6;
}

.stApp {
    background: __BACKDROP__;
    color: var(--starlight);
    font-family: 'Exo 2', sans-serif;
}

/* ---------- Living background ---------- */
/* Twinkling, drifting stars */
.stApp::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        radial-gradient(1.5px 1.5px at 20px 30px, #fff, transparent),
        radial-gradient(1px 1px at 90px 120px, #cfe9ff, transparent),
        radial-gradient(2px 2px at 160px 60px, #ffd6ea, transparent),
        radial-gradient(1px 1px at 230px 180px, #fff, transparent),
        radial-gradient(1.5px 1.5px at 310px 90px, #bff7f9, transparent);
    background-size: 340px 220px;
    animation: drift 90s linear infinite, twinkle 4s ease-in-out infinite alternate;
}
/* Aurora clouds slowly breathing */
.stApp::after {
    content: "";
    position: fixed; inset: -20%; pointer-events: none; z-index: 0;
    background:
        radial-gradient(40% 35% at 20% 25%, rgba(92, 225, 230, 0.20), transparent 70%),
        radial-gradient(35% 40% at 80% 30%, rgba(255, 111, 168, 0.18), transparent 70%),
        radial-gradient(45% 40% at 55% 85%, rgba(143, 123, 255, 0.22), transparent 70%);
    animation: aurora 26s ease-in-out infinite alternate;
}
@keyframes drift { to { background-position: -340px 220px; } }
@keyframes twinkle { from { opacity: 0.35; } to { opacity: 0.8; } }
@keyframes aurora {
    from { transform: translate3d(-4%, -3%, 0) scale(1); }
    to   { transform: translate3d(4%, 3%, 0) scale(1.12); }
}

/* Shooting stars */
.shooting {
    position: fixed; top: -10px; width: 2px; height: 110px; z-index: 0; pointer-events: none;
    background: linear-gradient(180deg, rgba(255,255,255,0), #fff);
    border-radius: 2px; opacity: 0;
    transform: rotate(35deg);
    animation: shoot 9s ease-in infinite;
}
.shooting.s1 { left: 22%; animation-delay: 1s; }
.shooting.s2 { left: 58%; animation-delay: 4.5s; animation-duration: 11s; }
.shooting.s3 { left: 84%; animation-delay: 7s; animation-duration: 13s; }
@keyframes shoot {
    0%   { opacity: 0; transform: translate(0, 0) rotate(35deg); }
    3%   { opacity: 1; }
    12%  { opacity: 0; transform: translate(-420px, 560px) rotate(35deg); }
    100% { opacity: 0; transform: translate(-420px, 560px) rotate(35deg); }
}

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1100px; position: relative; z-index: 1; }

/* ---------- Title ---------- */
.hero {
    display: flex; align-items: center; justify-content: center;
    gap: 22px; padding: 8px 0 4px; flex-wrap: wrap; text-align: center;
}
.hero img {
    height: 88px; width: auto; border-radius: 18px;
    animation: logo-pulse 3.2s ease-in-out infinite;
}
@keyframes logo-pulse {
    0%, 100% { box-shadow: 0 0 18px rgba(92, 225, 230, 0.45), 0 0 0 0 rgba(255, 111, 168, 0.0); }
    50%      { box-shadow: 0 0 42px rgba(255, 111, 168, 0.65), 0 0 70px rgba(92, 225, 230, 0.35); }
}
.hero h1 {
    font-family: 'Audiowide', sans-serif; font-weight: 400;
    font-size: clamp(2.4rem, 7vw, 4.4rem); letter-spacing: 0.1em;
    margin: 0; padding: 0;
    background: linear-gradient(90deg, #5ce1e6, #8f7bff, #ff6fa8, #ffb36f, #5ce1e6);
    background-size: 300% 100%;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: flow 7s linear infinite, glitch 1.4s steps(1) 1 both;
}
@keyframes flow { to { background-position: 300% 0; } }
/* One-time RGB-split glitch when the page loads */
@keyframes glitch {
    0%   { filter: drop-shadow(6px 0 #ff2d75) drop-shadow(-6px 0 #00f0ff); transform: translateX(-4px); }
    10%  { filter: drop-shadow(-5px 0 #ff2d75) drop-shadow(5px 0 #00f0ff); transform: translateX(5px) skewX(-8deg); }
    20%  { filter: drop-shadow(4px 0 #ff2d75) drop-shadow(-4px 0 #00f0ff); transform: translateX(-3px); }
    30%  { filter: drop-shadow(0 0 14px rgba(143,123,255,.7)); transform: none; }
    42%  { filter: drop-shadow(7px 0 #ff2d75) drop-shadow(-7px 0 #00f0ff); transform: translateX(3px) skewX(6deg); }
    52%, 100% { filter: drop-shadow(0 0 18px rgba(143, 123, 255, 0.6)); transform: none; }
}
.tagline {
    text-align: center; color: var(--dim); font-size: 1.15rem;
    margin: 8px 0 30px; font-weight: 600; letter-spacing: 0.02em;
}

/* ---------- Glass panels with a travelling neon edge ---------- */
.panel {
    position: relative;
    background: var(--panel);
    border-radius: 22px;
    padding: 24px 26px;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 12px 44px rgba(5, 3, 25, 0.5), inset 0 0 40px rgba(143, 123, 255, 0.07);
    animation: rise 0.7s cubic-bezier(.2,.8,.2,1) both;
}
.panel::before {
    content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 2px;
    background: conic-gradient(from var(--angle), transparent 0 55%, #5ce1e6 70%, #8f7bff 82%, #ff6fa8 92%, transparent 100%);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    animation: spin 5s linear infinite;
    pointer-events: none;
}
.panel.p2 { animation-delay: 0.15s; }
.panel.p3 { animation-delay: 0.3s; }
@keyframes spin { to { --angle: 360deg; } }
@keyframes rise { from { opacity: 0; transform: translateY(26px) scale(0.98); } to { opacity: 1; transform: none; } }
.panel h3 {
    font-family: 'Audiowide', sans-serif; font-weight: 400; font-size: 1.1rem;
    color: var(--starlight); margin: 0 0 14px; letter-spacing: 0.05em;
}

/* ---------- Uploader: breathing neon drop zone ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(26, 18, 70, 0.55);
    border: 2px dashed rgba(92, 225, 230, 0.65);
    border-radius: 22px; padding: 30px;
    animation: breathe 3s ease-in-out infinite;
    transition: transform 0.25s ease, border-color 0.25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    transform: scale(1.015); border-color: var(--sakura);
}
@keyframes breathe {
    0%, 100% { box-shadow: 0 0 10px rgba(92, 225, 230, 0.25), inset 0 0 18px rgba(92, 225, 230, 0.05); }
    50%      { box-shadow: 0 0 34px rgba(92, 225, 230, 0.55), inset 0 0 30px rgba(143, 123, 255, 0.14); }
}
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small { color: var(--starlight) !important; }
/* Keep Streamlit's icon glyphs on their own icon font so they never show as text */
[data-testid="stIconMaterial"], [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
.stButton > button, [data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, #5a43d6, #b04bd0, #ff6fa8);
    background-size: 200% 100%;
    color: #fff; border: none; border-radius: 12px; font-weight: 800; padding: 8px 20px;
    transition: background-position 0.4s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover, [data-testid="stFileUploaderDropzone"] button:hover {
    background-position: 100% 0; transform: translateY(-2px); color: #fff;
    box-shadow: 0 8px 24px rgba(255, 111, 168, 0.45);
}
:focus-visible { outline: 3px solid var(--aurora) !important; outline-offset: 2px; }

/* ---------- Images: laser scan + holographic tilt ---------- */
[data-testid="stImage"] { position: relative; overflow: hidden; border-radius: 16px; }
[data-testid="stImage"] img {
    border-radius: 16px; box-shadow: 0 8px 30px rgba(5, 3, 25, 0.55);
    transition: transform 0.4s ease, box-shadow 0.4s ease;
}
[data-testid="stImage"]:hover img {
    transform: perspective(800px) rotateY(-5deg) rotateX(3deg) scale(1.03);
    box-shadow: 0 18px 50px rgba(92, 225, 230, 0.35);
}
[data-testid="stImage"]::after {
    content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; pointer-events: none;
    background: linear-gradient(90deg, transparent, #5ce1e6, #fff, #5ce1e6, transparent);
    box-shadow: 0 0 18px 6px rgba(92, 225, 230, 0.7);
    animation: scan 1.8s ease-in-out 2 both;
}
@keyframes scan {
    0%   { top: 0; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

/* ---------- Result ---------- */
.result-label { color: var(--dim); font-size: 1rem; font-weight: 600; margin: 0; }
.result-title {
    font-family: 'Audiowide', sans-serif; font-weight: 400;
    font-size: clamp(1.5rem, 3.4vw, 2.3rem); line-height: 1.25;
    color: #fff; margin: 4px 0 18px;
    text-shadow: 0 0 24px rgba(92, 225, 230, 0.65);
    animation: reveal 1.1s cubic-bezier(.2,.8,.2,1) 0.2s both;
}
@keyframes reveal {
    from { opacity: 0; filter: blur(14px); letter-spacing: 0.35em; }
    to   { opacity: 1; filter: blur(0);    letter-spacing: normal; }
}
.confidence-wrap { display: flex; align-items: center; gap: 20px; margin-bottom: 6px; }
.confidence-ring {
    width: 112px; height: 112px; border-radius: 50%; flex: none;
    display: grid; place-items: center;
    background: conic-gradient(#5ce1e6, #8f7bff calc(var(--num) * 0.6%), #c06bff calc(var(--num) * 1%), rgba(255,255,255,0.14) 0);
    box-shadow: 0 0 34px rgba(160, 100, 255, 0.65);
    animation: count 1.8s cubic-bezier(.2,.8,.2,1) 0.3s both;
}
.confidence-ring > div {
    width: 88px; height: 88px; border-radius: 50%; background: #170f4a;
    display: grid; place-items: center;
    font-family: 'Audiowide', sans-serif; font-size: 1.35rem;
}
.confidence-ring > div::after { content: counter(num) "%"; counter-reset: num var(--num); }
.confidence-text strong { font-size: 1.25rem; display: block; }
.confidence-text span { color: var(--dim); font-size: 1rem; }

.facts { display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 16px; }
.fact {
    background: rgba(92, 225, 230, 0.10); border: 1px solid rgba(92, 225, 230, 0.4);
    border-radius: 12px; padding: 8px 14px;
}
.fact small { display: block; color: var(--dim); font-size: 0.8rem; }
.fact b { font-size: 1.05rem; }
.synopsis { line-height: 1.7; color: var(--starlight); max-width: 70ch; margin: 0; font-size: 1.02rem; }

/* ---------- Top guesses ---------- */
.bar-row { margin: 12px 0; }
.bar-head { display: flex; justify-content: space-between; gap: 12px; font-size: 1rem; margin-bottom: 5px; }
.bar-head span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { height: 10px; border-radius: 99px; background: rgba(255, 255, 255, 0.12); overflow: hidden; }
.bar-fill {
    height: 100%; border-radius: 99px; position: relative; overflow: hidden;
    background: linear-gradient(90deg, #5ce1e6, #8f7bff);
    animation: grow 1.2s cubic-bezier(.2,.8,.2,1) both;
}
.bar-fill::after {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(100deg, transparent 30%, rgba(255,255,255,0.65) 50%, transparent 70%);
    transform: translateX(-100%); animation: shine 2.6s ease-in-out 1.2s infinite;
}
.bar-row.top .bar-fill { background: linear-gradient(90deg, #ff6fa8, #ffb36f); }
.bar-row:nth-child(2) .bar-fill { animation-delay: 0.1s; }
.bar-row:nth-child(3) .bar-fill { animation-delay: 0.2s; }
.bar-row:nth-child(4) .bar-fill { animation-delay: 0.3s; }
.bar-row:nth-child(5) .bar-fill { animation-delay: 0.4s; }
@keyframes grow { from { width: 0; } }
@keyframes shine { to { transform: translateX(100%); } }

.hint { color: var(--dim); font-size: 1rem; text-align: center; margin-top: 16px; }
.footer { text-align: center; color: var(--dim); font-size: 0.9rem; margin: 40px 0 10px; }

[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stExpander"] {
    background: var(--panel); border: 1px solid rgba(160, 140, 255, 0.28); border-radius: 16px;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p, [data-testid="stExpander"] li { color: var(--starlight); }

/* ---------- Purple flash (confident results) ---------- */
.flash {
    position: fixed; inset: 0; z-index: 5; pointer-events: none;
    background: radial-gradient(circle at 50% 40%, rgba(176, 90, 255, 0.60), rgba(110, 60, 220, 0.28) 45%, transparent 78%);
    opacity: 0;
    animation: flash 1s ease-out 1 both;
}
@keyframes flash {
    0%   { opacity: 0; }
    12%  { opacity: 1; }
    100% { opacity: 0; }
}

/* Space above "How does this work?" */
[data-testid="stExpander"] { margin-top: 3rem; }

@media (prefers-reduced-motion: reduce) {
    .flash { display: none; }
    *, *::before, *::after { animation: none !important; transition: none !important; }
    .shooting { display: none; }
}
</style>
"""


def inject_styles():
    if os.path.exists(BACKGROUND_PATH):
        backdrop = (
            "linear-gradient(180deg, rgba(14,10,44,0.80), rgba(14,10,44,0.93)), "
            f"url('data:image/jpeg;base64,{file_to_base64(BACKGROUND_PATH)}') center / cover fixed no-repeat"
        )
    else:
        backdrop = (
            "radial-gradient(ellipse at 15% 10%, #3a2a8f 0%, transparent 45%), "
            "radial-gradient(ellipse at 85% 30%, #7a2c78 0%, transparent 40%), "
            "radial-gradient(ellipse at 50% 100%, #123a6b 0%, transparent 50%), "
            "#0e0a2c"
        )
    st.markdown(CSS.replace("__BACKDROP__", backdrop), unsafe_allow_html=True)


inject_styles()

# --- Header ---
logo_html = ""
if os.path.exists(LOGO_PATH):
    logo_html = f'<img src="data:image/png;base64,{file_to_base64(LOGO_PATH)}" alt="ANIMATRIX logo">'

st.markdown(
    html_block(
        '<div class="shooting s1"></div><div class="shooting s2"></div><div class="shooting s3"></div>',
        '<div class="hero">',
        logo_html,
        "<h1>ANIMATRIX</h1>",
        "</div>",
        '<p class="tagline">Upload a screenshot or poster and find out which anime it comes from.</p>',
    ),
    unsafe_allow_html=True,
)


# --- Data & model (loaded once, then cached) ---
@st.cache_data(show_spinner=False)
def load_metadata():
    return pd.read_csv(ANIME_METADATA_PATH)


class MyEfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models import efficientnet_b0
        self.network = efficientnet_b0(weights='IMAGENET1K_V1')
        self.network.classifier[1] = nn.Linear(self.network.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.network(x)


@st.cache_resource(show_spinner="Warming up the model...")
def load_model():
    m = MyEfficientNetB0(num_classes=len(ANIME_TITLES)).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

anime_metadata = load_metadata()
model = load_model()


def predict(image_file, top_k=5):
    """Returns the top matches as a list of (title, percent), best first."""
    img = Image.open(image_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.nn.functional.softmax(model(img_tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, min(top_k, len(ANIME_TITLES)))
    return [(ANIME_TITLES[i], v * 100) for v, i in zip(values.tolist(), indices.tolist())], img


def find_metadata(title):
    for candidate in (title, clean_title(title)):
        match = anime_metadata[anime_metadata["Name"] == candidate]
        if not match.empty:
            return match.iloc[0]
    return None


def confidence_words(pct):
    if pct >= 80:
        return "Very likely", "The model is quite sure about this one."
    if pct >= 50:
        return "Probably", "A good match, but check the other guesses too."
    return "Not sure", "Try a clearer image. See the other guesses below."


# --- Upload ---
uploaded_file = st.file_uploader(
    "Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    help="JPG or PNG. Clear scenes with characters work best.",
)

if uploaded_file is None:
    st.markdown(
        '<p class="hint">Nothing uploaded yet. Pick an image and the result appears here.</p>',
        unsafe_allow_html=True,
    )
else:
    try:
        with st.spinner("Scanning the galaxy of anime..."):
            matches, processed_img = predict(uploaded_file)
        best_title, best_pct = matches[0]
        shown_title = html.escape(clean_title(best_title))
        verdict, verdict_note = confidence_words(best_pct)

        info = find_metadata(best_title)
        if best_pct >= 80:
            st.markdown('<div class="flash"></div>', unsafe_allow_html=True)

        left, right = st.columns([1, 1.5], gap="large")

        with left:
            st.image(uploaded_file, caption="Your image", use_container_width=True)
            if info is not None and isinstance(info.get("Image URL"), str) and info["Image URL"].startswith("http"):
                st.image(info["Image URL"], caption="Official poster", width=220)

        with right:
            st.markdown(
                html_block(
                    f"<style>@keyframes count {{ to {{ --num: {round(best_pct)}; }} }}</style>",
                    '<div class="panel">',
                    '<p class="result-label">This looks like</p>',
                    f'<p class="result-title">{shown_title}</p>',
                    '<div class="confidence-wrap">',
                    '<div class="confidence-ring">',
                    "<div></div>",
                    "</div>",
                    f'<div class="confidence-text"><strong>{verdict}</strong><span>{verdict_note}</span></div>',
                    "</div>",
                    "</div>",
                ),
                unsafe_allow_html=True,
            )

            if info is not None:
                synopsis = info.get("Synopsis")
                synopsis = synopsis if isinstance(synopsis, str) else ""
                short = synopsis[:300] + "..." if len(synopsis) > 300 else synopsis
                rating = html.escape(str(info.get("Rating", "Unknown")))
                premiered = html.escape(str(info.get("Premiered", "Unknown")))
                st.markdown(
                    html_block(
                        '<div class="panel p2" style="margin-top:18px;">',
                        "<h3>About this anime</h3>",
                        '<div class="facts">',
                        f'<div class="fact"><small>Age rating</small><b>{rating}</b></div>',
                        f'<div class="fact"><small>Premiered</small><b>{premiered}</b></div>',
                        "</div>",
                        f'<p class="synopsis">{html.escape(short) if short else "No synopsis available."}</p>',
                        "</div>",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"No extra details found for {clean_title(best_title)} in the dataset.")

            rows = []
            for i, (title, pct) in enumerate(matches):
                top_class = " top" if i == 0 else ""
                rows.append(
                    html_block(
                        f'<div class="bar-row{top_class}">',
                        f'<div class="bar-head"><span>{html.escape(clean_title(title))}</span><span>{pct:.1f}%</span></div>',
                        f'<div class="bar-track"><div class="bar-fill" style="width:{max(pct, 1):.1f}%;"></div></div>',
                        "</div>",
                    )
                )
            st.markdown(
                '<div class="panel p3" style="margin-top:18px;"><h3>Top 5 guesses</h3>'
                + "".join(rows)
                + "</div>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"We couldn't read that image. Try a different JPG or PNG. Details: {e}")

with st.expander("How does this work?"):
    st.markdown(
        """
        - The app compares your image with what it learned from **100 popular anime**.
        - It shows its best guess, how confident it is, and four runner-up guesses.
        - Results are best for screenshots and key art. Blurry, tiny, or heavily cropped images may confuse it.
        """
    )

st.markdown('<p class="footer">ANIMATRIX · powered by EfficientNet-B0</p>', unsafe_allow_html=True)