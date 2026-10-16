"""Deterministic syntax cross-check, report only; never relabel or remove records.

usage: python -m eval.oracle_ground_truth [PREFIX]  -> reports/oracle_ground_truth_<prefix>.json
Syntax presence is evidence to inspect, not proof of exploitability or a wrong benign label.
"""
import argparse
import base64
import binascii
import csv
import html
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, unquote

ROOT = Path(__file__).resolve().parent.parent
JWT = re.compile(r'(?<![\w.-])([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]*)(?![\w.-])')
HEADER = re.compile(r'[\r\n]+[ \t]*(?:[!#$%&\'*+.^_`|~0-9A-Za-z-]+[ \t]*:|HTTP/\d)')  # injected header or status line
PROTO = re.compile(r'(?:^|[.\[])__proto__(?:$|[.\[\]])|(?:^|[.\[])constructor(?:\.|\]\[|\[)prototype(?:$|[.\[\]])')


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from strings(value)


def decode(value):
    for _ in range(3):
        new = html.unescape(unquote(value))
        if new == value:
            break
        value = new
    return value


def unbase64(value):
    return base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True).decode('utf-8')


def views(value):
    out = list(dict.fromkeys((value, decode(value))))
    for text in tuple(out):
        token = text.strip()
        if token.lower().startswith('base64:'):
            token = token[7:]
        if len(token) < 8 or not re.fullmatch(r'[A-Za-z0-9_+/=-]+', token):
            continue
        try:
            decoded = decode(unbase64(token))
        except (ValueError, UnicodeError, binascii.Error):
            continue
        if decoded and all(c.isprintable() or c in '\t\r\n\0' for c in decoded):
            out.append(decoded)
    return list(dict.fromkeys(out))


def field_values(text):
    yield text
    try:
        yield from strings(json.loads(text))
    except (ValueError, RecursionError):
        pass
    if '=' in text:
        query = text.split('?', 1)[-1]
        for _, value in parse_qsl(query, keep_blank_values=True):
            yield value
    if ',' in text or '\n' in text:
        try:
            for row in csv.reader(io.StringIO(text)):
                yield from row
        except csv.Error:
            pass


def jwt_auth_bypass(tel, texts):
    context = decode(json.dumps(tel.get('context') or {})).lower()
    public = ('public' in context and bool(re.search(r'\b(?:rsa|ec|ecdsa)\b', context))) or \
             bool(re.search(r'begin (?:rsa |ec )?public key', context)) or \
             bool(re.search(r'allowed_algs"\s*:\s*\[\s*"(?:rs|es|ps)\d', context))  # verifier pinned to asymmetric algs
    def bad(header, signed=True):
        alg, kid = str(header.get('alg') or '').lower(), str(header.get('kid', ''))
        return (alg == 'none' or any(k in header for k in ('jku', 'x5u', 'jwk')) or
                bool(re.search(r'\.\.[/\\]|[\'";|&`$<>\r\n]|\b(?:union|select|or|and)\b', decode(kid), re.I)) or
                (alg.startswith('hs') and public) or not signed)
    for text in texts:
        for match in JWT.finditer(text):
            try:
                header, payload = (json.loads(unbase64(s)) for s in match.group(1, 2))
            except (ValueError, UnicodeError, binascii.Error, RecursionError):
                continue
            if isinstance(header, dict) and isinstance(payload, dict) and bad(header, bool(match.group(3))):
                return True
    # records may carry only the decoded-from header segment (e.g. context.token_material.header_b64)
    for s in strings(tel):
        if s.startswith('eyJ') and '.' not in s:
            try:
                header = json.loads(unbase64(s))
            except (ValueError, UnicodeError, binascii.Error, RecursionError):
                continue
            if isinstance(header, dict) and 'alg' in header and bad(header):
                return True
    return False


def xxe(tel, texts):
    for raw in strings((tel.get('http') or {}).get('body')):
        for body in views(raw):
            for subset in re.findall(r'<!DOCTYPE\s+[^\[>]+\[([\s\S]*?)\]\s*>', body, re.I):
                if re.search(r'<!ENTITY\s+(?:%\s+)?[^\s>]+\s+(?:SYSTEM|PUBLIC)\s+[\'"]', subset, re.I):
                    return True
            namespaces = re.findall(r'xmlns(?::([\w.-]+))?\s*=\s*[\'"]http://www.w3.org/2001/XInclude[\'"]', body)
            if any(re.search(r'<\s*' + (re.escape(ns) + ':' if ns else '') + r'include\b', body)
                   for ns in namespaces):
                return True
    return False


def tokens(text):
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif c.isalnum() or c in '_-':
            start = i
            while i < len(text) and (text[i].isalnum() or text[i] in '_-'):
                i += 1
            out.append(text[start:i].lower())
        else:
            out.append(c)
            i += 1
    return out


def ldap_xpath_injection(tel, texts):
    # request input only: context carries the application's own filter template, e.g. (&(objectClass=x)(uid={value}))
    http = tel.get('http') or {}
    grab = lambda keys: list(dict.fromkeys(v for raw in strings({k: http.get(k) for k in keys})
                                           for field in field_values(raw) for v in views(field)))
    texts = grab(('target', 'body', 'headers', 'cookies', 'decoded_views'))
    if any('\0' in t for t in grab(('target', 'headers', 'cookies'))):  # NUL truncation; binary bodies carry bare NULs
        return True
    for text in texts:
        if any(s in text for s in (')(', '*)(', '|(', '&(')) or re.search(r'[)*]\x00', text):
            return True
        ts = tokens(text)
        for i, token in enumerate(ts):
            tail = ts[i:]
            if token in ('count', 'name', 'string-length') and tail[1:2] == ['(']:
                return True
            if tail[:4] == [']', '|', '/', '/']:
                return True
            if token not in ('or', 'and'):
                continue
            # The request may close a quote opened by the application's XPath expression.
            j = i + 1
            if j >= len(ts) or ts[j] not in ("'", '"'):
                continue
            quote = ts[j]; j += 1
            while j < len(ts) and ts[j] != quote:
                j += 1
            if ts[j:j+3] == [quote, '=', quote]:
                return True
    return False


def csv_formula_injection(tel, texts):
    for text in texts:
        for value in field_values(text):
            value = re.sub(r'^[\s\'\"]+', '', decode(value))
            if not value or value[0] not in '=+-@':
                continue
            expr = value[1:].lstrip()
            if re.fullmatch(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\d{2,4}[-/]\d{1,2}[-/]\d{1,4}', expr):
                continue
            if (re.match(r'(?:[A-Za-z_][\w.]*\s*\(|\$?[A-Za-z]{1,3}\$?[1-9]\d*\b|cmd\s*\|)', expr, re.I) or
                    re.match(r'[A-Za-z_][\w.]*\s*\|.+!\$?[A-Za-z]+\$?\d+', expr)):
                return True
    return False


def pairs(obj):
    return obj.items() if isinstance(obj, dict) else (obj or [])


def crlf_header_injection(tel, texts):
    http = tel.get('http') or {}
    values = [(str(http.get('protocol', '')).upper() in ('HTTP/2', 'HTTP/2.0', 'H2'), http.get('target') or '')]
    for name in ('headers', 'pseudo_headers'):
        values += [(str(k).startswith(':'), v) for k, v in pairs(http.get(name)) if isinstance(v, str)]
    values += [(False, d.get('value')) for d in http.get('decoded_views') or [] if isinstance(d, dict) and isinstance(d.get('value'), str)
               and not str(d.get('field', '')).startswith('body')]  # chunked/multipart bodies carry CRLF framing
    return any((pseudo and ('\r' in value or '\n' in value)) or bool(HEADER.search(value))
               for pseudo, raw in values for value in views(raw))


def polluted(obj, path=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = path + '.' + decode(key) if path else decode(key)
            if PROTO.search(child) or polluted(value, child):
                return True
    elif isinstance(obj, list):
        return any(polluted(value, path) for value in obj)
    return False


def prototype_pollution(tel, texts):
    http = tel.get('http') or {}
    keys = [str(k) for k, _ in pairs(http.get('query'))]
    target = http.get('target') or ''
    if '?' in target:
        keys += [k for k, _ in parse_qsl(decode(target).split('?', 1)[1], keep_blank_values=True)]
    for raw in strings(http.get('body')):
        for body in views(raw):
            try:
                if polluted(json.loads(body)):
                    return True
            except (ValueError, RecursionError):
                pass
            if '=' in body and not body.lstrip().startswith('{'):
                keys += [k for k, _ in parse_qsl(body, keep_blank_values=True)]
    return any(PROTO.search(decode(key)) is not None for key in keys)


CHECKS = {f.__name__: f for f in (jwt_auth_bypass, xxe, ldap_xpath_injection, csv_formula_injection,
                                  crlf_header_injection, prototype_pollution)}


def checks(tel):
    texts = list(dict.fromkeys(v for raw in strings(tel) for field in field_values(raw) for v in views(field)))
    return {name: check(tel, texts) for name, check in CHECKS.items()}


def main(prefix='fast_'):
    if not re.fullmatch(r'[A-Za-z0-9_.-]*', prefix) or prefix in ('.', '..'):
        raise ValueError('prefix must be a filename prefix')
    files = sorted((ROOT / 'datasets_ready').glob(f'{prefix}*.jsonl'))
    if not files:
        raise FileNotFoundError('no matching dataset files')
    stats = {name: {'n': 0, 'pass': 0, 'fail': 0, 'pass_rate': 0.0, 'failing_ids': []} for name in CHECKS}
    benign = {'ordinary': {}, 'hard_negative': {}}
    provenance = defaultdict(Counter)
    for fn in files:
        with fn.open() as source:
            for line in source:
                record = json.loads(line)
                meta = record['metadata']; cls = meta['threat_type']
                if cls not in CHECKS and not cls.startswith('benign_'):
                    continue
                content = next(m['content'] for m in record['messages'] if m['role'] == 'user')
                tel = json.loads(content.split('] ', 1)[-1])
                fired = checks(tel)
                origin = meta.get('provenance', 'unknown')
                generator = meta.get('generator', 'unknown')
                provenance[cls][f'{origin}/{generator}'] += 1
                if cls in CHECKS:
                    row = stats[cls]; row['n'] += 1
                    row['pass' if fired[cls] else 'fail'] += 1
                    if not fired[cls] and len(row['failing_ids']) < 5:
                        row['failing_ids'].append(record['id'])
                else:
                    group = 'hard_negative' if cls.startswith('benign_hard_negative_') else 'ordinary'
                    row = benign[group].setdefault(cls, {'n': 0, 'any_hit': 0, 'checks': dict.fromkeys(CHECKS, 0)})
                    row['n'] += 1; row['any_hit'] += int(any(fired.values()))
                    for name, hit in fired.items():
                        row['checks'][name] += int(hit)
    for row in stats.values():
        row['pass_rate'] = round(row['pass'] / row['n'], 6) if row['n'] else 0.0
    report = {'prefix': prefix, 'report_only': True, 'classes': stats, 'benign_audit': benign,
              'provenance_generator': {c: dict(sorted(v.items())) for c, v in sorted(provenance.items())}}
    dest = ROOT / 'reports' / f'oracle_ground_truth_{prefix}.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report, indent=1) + '\n')
    print('class                              n   pass   fail  pass_rate')
    for cls, row in stats.items():
        print(f"{cls:30s} {row['n']:5d} {row['pass']:6d} {row['fail']:6d} {row['pass_rate']:9.2%}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix', nargs='?', default='fast_')
    main(ap.parse_args().prefix)
