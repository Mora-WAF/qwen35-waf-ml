import base64
import copy
import json
from urllib.parse import quote

import pytest

from eval.oracle_ground_truth import checks, main


def b64(value):
    return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip('=')


def jwt(header, payload=None, signature='c2ln'):
    return f'{b64(header)}.{b64(payload or {"sub": "u"})}.{signature}'


def tel(value='', **http):
    return {'http': {'target': '/', 'headers': [], 'body': value, **http}}


@pytest.mark.parametrize('header,signature,context,expected', [
    ({'alg': 'nOnE'}, 'c2ln', {}, True),
    ({'alg': 'RS256', 'jku': 'https://keys.test'}, 'c2ln', {}, True),
    ({'alg': 'RS256', 'x5u': 'https://keys.test'}, 'c2ln', {}, True),
    ({'alg': 'RS256', 'jwk': {}}, 'c2ln', {}, True),
    ({'alg': 'HS256', 'kid': '../../key'}, 'c2ln', {}, True),
    ({'alg': 'HS256', 'kid': "x' OR 1=1--"}, 'c2ln', {}, True),
    ({'alg': 'HS256'}, 'c2ln', {'key_type': 'RSA public key'}, True),
    ({'alg': 'HS384'}, 'c2ln', {'key_type': 'EC', 'key_use': 'public'}, True),
    ({'alg': 'RS256'}, '', {}, True),
    ({'alg': 'HS256'}, 'c2ln', {}, False),
    ({'alg': 'RS256', 'kid': 'key-123'}, 'c2ln', {}, False),
])
def test_jwt(header, signature, context, expected):
    t = tel(headers=[['Authorization', 'Bearer ' + jwt(header, signature=signature)]])
    t['context'] = context
    assert checks(t)['jwt_auth_bypass'] is expected


@pytest.mark.parametrize('value', ['no token', 'abc.def.', 'e30.W10.', jwt({'alg': 'RS256'}) + '!'])
def test_jwt_negative(value):
    assert not checks(tel(value))['jwt_auth_bypass']


@pytest.mark.parametrize('body,expected', [
    ('<!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]><r>&x;</r>', True),
    ('<!DOCTYPE r [<!ENTITY % x PUBLIC "id" "https://x.test">%x;]><r/>', True),
    ('<r xmlns:xi="http://www.w3.org/2001/XInclude"><xi:include href="file:///x"/></r>', True),
    ('<r xmlns:q="http://www.w3.org/2001/XInclude"><q:include href="file:///x"/></r>', True),
    ('<!DOCTYPE r [<!ENTITY x "ordinary">]><r/>', False),
    ('<r>SYSTEM PUBLIC XInclude</r>', False),
    ('<!ENTITY x SYSTEM "file:///x"><r/>', False),
])
def test_xxe(body, expected):
    assert checks(tel(body))['xxe'] is expected
    assert not checks(tel(headers=[['X-Note', body]]))['xxe']


@pytest.mark.parametrize('value,expected', [
    ('*)(uid=*)', True), (')(', True), ('|(uid=a)', True), ('&(uid=a)', True), ('a\0b', True),
    ("' or '1'='1", True), ("' or ''='", True), ('count(//user)', True),
    ('name(/*)', True), ('string-length(//x)', True), (']|//user', True),
    ('ordinary (notes)', False), ('color or size', False), ("O'Reilly", False),
    ('count items', False), ("or '1'=plain", False),
])
def test_ldap_xpath(value, expected):
    assert checks(tel(target='/?q=' + quote(value)))['ldap_xpath_injection'] is expected


@pytest.mark.parametrize('value,expected', [
    ('=SUM(A1:A9)', True), ('\t\r\"+SUM(A1)', True), ('-A12', True), ('@cmd|\'/C calc\'!A0', True),
    ('=DDE("cmd","x","y")', True), ('-123.45', False), ('+1e3', False),
    ('-2026-01-01', False), ('@ordinary', False), ('text =SUM(A1)', False),
])
def test_csv(value, expected):
    assert checks(tel(json.dumps({'name': value})))['csv_formula_injection'] is expected
    assert checks(tel(headers=[['X-Field', value]]))['csv_formula_injection'] is expected


def test_csv_fields():
    assert checks(tel('name,value\nuser,=SUM(A1:A2)'))['csv_formula_injection']
    assert checks(tel('name=%2BSUM%28A1%29'))['csv_formula_injection']
    assert not checks(tel('name,value\nuser,-12.5'))['csv_formula_injection']


@pytest.mark.parametrize('http,expected', [
    ({'headers': [['X-Test', 'a%250d%250aSet-Cookie: x=1']]}, True),
    ({'target': '/a%0aX-Test:%20yes'}, True),
    ({'headers': [[':path', '/a\nanything']]}, True),
    ({'pseudo_headers': {':authority': 'a\nb'}}, True),
    ({'headers': [['X-Test', 'a\nordinary']]}, False),
    ({'body': 'a\r\nX-Test: yes'}, False),
])
def test_crlf(http, expected):
    assert checks(tel(**http))['crlf_header_injection'] is expected


@pytest.mark.parametrize('http,expected', [
    ({'body': '{"a":[{"__proto__":{"admin":true}}]}'}, True),
    ({'body': '{"constructor":{"prototype":{"admin":true}}}'}, True),
    ({'body': '{"constructor[prototype]":{"admin":true}}'}, True),
    ({'target': '/?a%255B__proto__%255D=x'}, True),
    ({'query': [['constructor.prototype.admin', 'true']]}, True),
    ({'body': 'x[constructor][prototype][admin]=true'}, True),
    ({'body': '{"note":"__proto__"}'}, False),
    ({'body': '{"my__proto__key":1}'}, False),
    ({'target': '/?note=__proto__'}, False),
    ({'body': '{"constructor":{"name":"prototype"}}'}, False),
])
def test_prototype(http, expected):
    assert checks(tel(**http))['prototype_pollution'] is expected


def test_decoding_and_context():
    xml = '<!DOCTYPE r [<!ENTITY x SYSTEM "file:///x">]><r/>'
    assert checks(tel(quote(quote(quote(xml)))))['xxe']
    assert checks(tel(xml.replace('<', '&lt;').replace('>', '&gt;')))['xxe']
    assert checks(tel(base64.urlsafe_b64encode(xml.encode()).decode()))['xxe']
    t = tel(); t['context'] = {'input': jwt({'alg': 'none'})}
    assert checks(t)['jwt_auth_bypass']
    t['context'] = {'input': '=SUM(A1)'}
    assert checks(t)['csv_formula_injection']


def test_report(tmp_path, monkeypatch):
    monkeypatch.setattr('eval.oracle_ground_truth.ROOT', tmp_path)
    source = tmp_path / 'datasets_ready'; source.mkdir()
    rows = []
    for i, cls in enumerate(['jwt_auth_bypass'] * 7 + ['benign_standard_traffic',
                           'benign_hard_negative_dev', 'benign_high_rate_legit']):
        t = tel('plain' if i < 7 else jwt({'alg': 'none'}))
        rows.append({'id': f'record-{i}', 'metadata': {'threat_type': cls, 'provenance': 'synthetic',
                     'generator': 'test'}, 'messages': [{'role': 'user', 'content': '[L7_HTTP] ' + json.dumps(t)}]})
    before = copy.deepcopy(rows)
    path = source / 'fast_test.jsonl'; path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    raw = path.read_bytes()
    report = main('fast_')
    assert report['classes']['jwt_auth_bypass'] == {'n': 7, 'pass': 0, 'fail': 7, 'pass_rate': 0.0,
                                                  'failing_ids': [f'record-{i}' for i in range(5)]}
    for group, cls in [('ordinary', 'benign_standard_traffic'), ('ordinary', 'benign_high_rate_legit'),
                       ('hard_negative', 'benign_hard_negative_dev')]:
        audit = report['benign_audit'][group][cls]
        assert audit['n'] == audit['any_hit'] == audit['checks']['jwt_auth_bypass'] == 1
    assert json.loads((tmp_path / 'reports/oracle_ground_truth_fast_.json').read_text()) == report
    assert path.read_bytes() == raw and rows == before
    assert 'plain' not in json.dumps(report) and 'eyJ' not in json.dumps(report)


def test_additional_formula_and_token_edges():
    assert not checks(tel(jwt({'alg': None})))['jwt_auth_bypass']
    assert checks(tel('\u2003\"=SUM(A1)'))['csv_formula_injection']
    assert checks(tel("=MSEXCEL|'sheet'!A1"))['csv_formula_injection']
    assert not checks(tel('=ordinary|text'))['csv_formula_injection']
    assert checks(tel('[__proto__][admin]=true'))['prototype_pollution']


def test_http2_path_is_pseudo_header():
    assert checks(tel(protocol='HTTP/2', target='/a%0aanything'))['crlf_header_injection']
    assert not checks(tel(protocol='HTTP/1.1', target='/a%0aanything'))['crlf_header_injection']
