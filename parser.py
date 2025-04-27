import tempfile
import os
import shutil
import gzip
import re
import json
import logging
from typing import List, Dict, Any
from git import Repo
from dotenv import load_dotenv
from pydantic import ValidationError
from google import genai
from google.genai import types
from schemas import BaseIssue, EnrichedIssue

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # set to DEBUG to capture file-level logs
logger = logging.getLogger(__name__)

# Load environment and configure Gemini
load_dotenv()
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# File extensions and domains
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif', '.svg')
CODE_EXTS = ('.js', '.jsx', '.ts', '.tsx', '.css', '.html')
MODERN_IMAGE_EXTS = ('.webp', '.avif')
THIRD_PARTY_DOMAINS = [
    'googleapis.com', 'gstatic.com', 'facebook.net',
    'analytics.com', 'hotjar.com'
]

# Impact weights for sorting (static guidelines)
IMPACT_WEIGHTS = {
    'Use efficient image formats': 3,
    'Optimize image dimensions': 3,
    'Lazy-load offscreen images': 2,
    'Minify CSS': 1,
    'Minify JavaScript': 1,
    'Remove unused CSS': 2,
    'Remove unused JavaScript': 2,
    'Enable text compression': 2,
    'Use HTTP caching': 2,
    'Preload critical assets': 2,
    'Defer non-critical CSS': 2,
    'Defer non-critical JavaScript': 2,
    'Use responsive images': 2,
    'Use modern image formats': 3,
    'Limit third-party scripts': 2,
    'Preconnect to required origins': 2,
}

# All guidelines
GUIDELINES = [
    'Use efficient image formats', 'Optimize image dimensions',
    'Lazy-load offscreen images', 'Minify CSS', 'Minify JavaScript',
    'Remove unused CSS', 'Remove unused JavaScript',
    'Enable text compression', 'Use HTTP caching', 'Preload critical assets',
    'Defer non-critical CSS', 'Defer non-critical JavaScript',
    'Use responsive images', 'Use modern image formats',
    'Reduce server response times', 'Enable GZIP compression',
    'Use a CDN for static assets', 'Limit third-party scripts',
    'Preconnect to required origins', 'Reduce DOM size',
    'Avoid synchronous layouts', 'Use efficient CSS selectors',
    'Avoid long tasks', 'Use web workers for expensive tasks'
]

# Split guidelines
STATIC_GUIDELINES = list(IMPACT_WEIGHTS.keys())
LLM_GUIDELINES = [g for g in GUIDELINES if g not in STATIC_GUIDELINES]


def clone_repo(repo_url: str) -> str:
    """Clone the repository to a temp directory and return its path."""
    logger.info('Cloning repository: %s', repo_url)
    tmpdir = tempfile.mkdtemp()
    Repo.clone_from(repo_url, tmpdir)
    return tmpdir

# JSON cleanup helper
def cleanup_json_text(text: str) -> str:
    try:
        start, end = text.find('['), text.rfind(']')
        text = text[start:end+1]
    except Exception:
        pass
    text = re.sub(r"\bNone\b", 'null', text)
    text = re.sub(r"\bTrue\b", 'true', text)
    text = re.sub(r"\bFalse\b", 'false', text)
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    return text

# Single Gemini call
def call_gemini(prompt: str, retries: int = 2) -> Any:
    for i in range(retries):
        resp = client.models.generate_content(
            model='gemini-2.5-flash-preview-04-17',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        text = resp.text
        logger.debug('Gemini response: %s', text)
        try:
            return json.loads(cleanup_json_text(text))
        except json.JSONDecodeError as e:
            logger.warning('Invalid JSON on attempt %d: %s', i+1, e)
            if i < retries - 1:
                prompt += '\nReminder: Return valid JSON only.'
                continue
            return None
    return None

# Compute metrics
def compute_metrics(root_dir: str) -> Dict[str, Any]:
    logger.info('Computing metrics in %s', root_dir)
    metrics = {
        'total_bytes': 0, 'image_bytes': 0,
        'js_bytes': 0, 'code_bytes': 0,
        'third_party_requests': 0, 'uncompressed_assets': []
    }
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(dirpath, f)
            logger.debug('Analyzing file for metrics: %s', path)
            ext = os.path.splitext(f)[1].lower()
            try:
                raw = open(path, 'rb').read()
                size = len(gzip.compress(raw))
            except Exception:
                size = os.path.getsize(path)
            metrics['total_bytes'] += size
            if ext in IMAGE_EXTS:
                metrics['image_bytes'] += size
            if ext in ('.js', '.css'):
                metrics['js_bytes'] += size
            if ext in CODE_EXTS:
                metrics['code_bytes'] += size
            if ext in ('.js', '.css') and size > 1024:
                metrics['uncompressed_assets'].append(os.path.relpath(path, root_dir))
            if any(domain in path for domain in THIRD_PARTY_DOMAINS):
                metrics['third_party_requests'] += 1
    logger.info('Metrics computed: %s', metrics)
    return metrics

# Static guidelines checks
def check_guidelines_static(root_dir: str) -> List[Dict[str, Any]]:
    logger.info('Running static checks in %s', root_dir)
    issues: List[Dict[str, Any]] = []
    all_code = ''
    file_contents: List[Dict[str, Any]] = []
    # determine project root and collect code files with context
    for dirpath, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules')]
        for f in files:
            path = os.path.join(dirpath, f)
            ext = os.path.splitext(f)[1].lower()
            # log scanning of each relevant code/HTML/CSS/JS/TSX file
            if ext in CODE_EXTS:
                logger.debug('Scanning code file for static rules: %s', path)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as cf:
                        content = cf.read()
                    # store per-file for precise issue attribution
                    file_contents.append({'path': os.path.relpath(path, root_dir), 'content': content})
                    all_code += content + ''
                except Exception:
                    logger.warning('Failed to read file for static analysis: %s', path)
    
    # 1) Unminified assets: JS/TS and CSS
    logger.debug('Checking unminified JS/TS/CSS assets')
    for file in file_contents:
        path = file['path']
        content = file['content']
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.js', '.jsx', '.ts', '.tsx'):
            if len(content) > 1024 and content.count('') < len(content)/200 and not path.endswith(('.min.js',)):
                logger.info('Issue: Unminified JavaScript detected in %s', path)
                issues.append({'type': 'Minify JavaScript', 'file': path})
        if ext == '.css':
            if len(content) > 1024 and content.count('') < len(content)/200 and not path.endswith(('.min.css',)):
                logger.info('Issue: Unminified CSS detected in %s', path)
                issues.append({'type': 'Minify CSS', 'file': path})
    
    # 2) Missing alt attributes per file
    logger.debug('Checking for missing alt attributes in HTML/JSX/TSX files')
    for file in file_contents:
        path = file['path']
        if path.endswith(('.html','.jsx','.tsx')):
            if re.search(r'<img(?![^>]*alt=)[^>]*>', file['content'], re.IGNORECASE):
                logger.info('Issue: Missing alt attribute in %s', path)
                issues.append({'type': 'MissingAltAttr', 'file': path})
    
    # 3) Render-blocking resources
    logger.debug('Checking for render-blocking scripts and CSS per file')
    for file in file_contents:
        path = file['path']
        content = file['content']
        if re.search(r'<script(?![^>]*(async|defer))[^>]+src=', content, re.IGNORECASE):
            logger.info('Issue: Render-blocking script in %s', path)
            issues.append({'type': 'RenderBlockingScript', 'file': path})
        if re.search(r'<link[^>]+rel="stylesheet"', content, re.IGNORECASE) and not re.search(r'<link[^>]+rel="stylesheet"[^>]*media=', content, re.IGNORECASE):
            logger.info('Issue: Render-blocking CSS in %s', path)
            issues.append({'type': 'RenderBlockingCSS', 'file': path})
    
    # 4) Missing cache policy meta per file
    logger.debug('Checking for cache-control meta tags per file')
    for file in file_contents:
        path = file['path']
        if re.search(r'<meta[^>]+http-equiv=["\']Cache-Control["\']', file['content'], re.IGNORECASE) is None:
            logger.info('Issue: Missing cache policy meta tag in %s', path)
            issues.append({'type': 'MissingCachePolicy', 'file': path})
    
    # 5) No code splitting (dynamic import) per aggregated code
    logger.debug('Checking for dynamic import() usage')
    if 'import(' not in all_code:
        logger.info('Issue: NoCodeSplitting detected')
        issues.append({'type': 'NoCodeSplitting', 'file': None})
    
    # 6) Image-specific checks: legacy formats and dimensions
    logger.debug('Checking image files for format and size issues')
    try:
        from PIL import Image
        for dirpath, _, files in os.walk(root_dir):
            for f in files:
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, root_dir)
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTS:
                    logger.debug('Analyzing image file: %s', path)
                    if ext not in MODERN_IMAGE_EXTS:
                        logger.info('Issue: Inefficient image format %s', rel)
                        issues.append({'type': 'LegacyImageFormat', 'file': rel})
                    try:
                        img = Image.open(path)
                        w, h = img.size
                        size = os.path.getsize(path)
                        if w*h > 1_000_000 and size > 200_000:
                            logger.info('Issue: Oversized image %s', rel)
                            issues.append({'type': 'OversizedImage', 'file': rel})
                    except Exception:
                        pass
    except ImportError:
        logger.warning('Pillow not installed; skipping image dimension checks')
    
    logger.info('Static checks found %d issues', len(issues))
    return issues

# Batched LLM checks
def check_guidelines_llm_batched(root_dir: str) -> List[Dict[str, Any]]:
    logger.info('Running batched LLM checks in %s', root_dir)
    code_snippet = ''
    for dirpath, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules')]
        for f in files:
            path = os.path.join(dirpath, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in CODE_EXTS + IMAGE_EXTS:
                logger.debug('Including file in LLM snippet: %s', path)
                try:
                    code_snippet += open(path, 'r', encoding='utf-8', errors='ignore').read()[:10000]
                except:
                    pass
    bullet_list = '\n'.join(f"- {g}" for g in LLM_GUIDELINES)
    prompt = (
        "You are a web performance auditor. Given the repository code below, "
        "evaluate compliance with these sustainable web design guidelines.\n\n"
        "Guidelines:\n" + bullet_list + "\n\n"
        "Code Sample (truncated):\n```\n" + code_snippet + "\n```\n\n"
        "Respond ONLY with a JSON array: [ { 'type': string, 'compliant': boolean, 'explanation': string } ]"
    )
    resp = call_gemini(prompt)
    results = []
    if isinstance(resp, list):
        for r in resp:
            if not r.get('compliant', True):
                logger.info('LLM reported non-compliance: %s', r['type'])
                results.append({'type': r['type'], 'file': None})
    logger.info('LLM checks found %d issues', len(results))
    return results

# Enrich all issues in one LLM call
def enrich_all_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info('Enriching %d issues via LLM', len(issues))
    if not issues:
        return []
    try:
        validated = [BaseIssue(type=i['type'], file=i.get('file')).model_dump() for i in issues]
    except ValidationError:
        logger.error('Issue validation failed; returning raw issues')
        return issues
    prompt = (
        "You are a sustainability assistant. For each issue below, return a JSON array of objects "
        "with keys: type, file, severity, impact, solution.\n" + json.dumps(validated, indent=2)
    )
    resp = call_gemini(prompt)
    gemini_map = {(g['type'], g.get('file')): g for g in (resp if isinstance(resp, list) else [])}
    enriched = []
    for base in validated:
        key = (base['type'], base.get('file'))
        meta = gemini_map.get(key, {})
        issue = {
            'type': base['type'], 'file': base.get('file'),
            'severity': meta.get('severity', 'Medium'),
            'impact': meta.get('impact', 'Contributes to energy consumption'),
            'solution': meta.get('solution', 'Refer to guidelines'),
            'weight': IMPACT_WEIGHTS.get(base['type'], 1)
        }
        logger.info('Enriched issue: %s => severity=%s', issue['type'], issue['severity'])
        try:
            enriched.append(EnrichedIssue(**issue).model_dump())
        except ValidationError:
            enriched.append(issue)
    logger.info('Enrichment completed: %d issues', len(enriched))
    return enriched

# Main parser
def parse(repo_url: str) -> Dict[str, Any]:
    logger.info('--- Starting analysis for %s ---', repo_url)
    repo_dir = clone_repo(repo_url)
    # Determine project root: client/ or frontend/ if present
    base_dir = repo_dir
    for sub in ('client', 'frontend'):
        subdir = os.path.join(repo_dir, sub)
        if os.path.isdir(subdir):
            logger.info('Detected subfolder "%s", using as root', sub)
            base_dir = subdir
            break
    try:
        metrics = compute_metrics(base_dir)
        static_issues = check_guidelines_static(base_dir)
        llm_issues = check_guidelines_llm_batched(base_dir)
        all_issues = static_issues + llm_issues
        enriched = enrich_all_issues(all_issues)
        enriched.sort(key=lambda x: (-x.get('weight', 0), x.get('severity', 'Medium')))
        logger.info('--- Analysis complete: Found %d total issues ---', len(enriched))
        return {'metrics': metrics, 'issues': enriched}
    finally:
        shutil.rmtree(repo_dir)
        logger.info('Cleaned up temp directory')
