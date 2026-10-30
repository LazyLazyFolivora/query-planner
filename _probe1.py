import json, pathlib
from src.kg import ToyKG
from src.pipeline import build_pipeline
from src.schema import Question, RequirementGroup, Triple

RULE = dict(parser='rule', encoder='feature', stop='rule', selector='heuristic')

root = pathlib.Path('.')
kg = ToyKG.from_json(root / 'data' / 'toy_kg.json')
pipe = build_pipeline(kg, **RULE)
qs = json.loads((root / 'data' / 'questions.json').read_text(encoding='utf-8'))['questions']
for q in qs:
    groups = [RequirementGroup(g['gid'], tuple(Triple(a['head'], a['relation'], a['tail']) for a in g['any_of']), float(g.get('weight', 1.0))) for g in q['requirement_groups']]
    Q = Question(qid=q['qid'], question=q['question'], qtype=q['qtype'], budget=dict(q['budget']),
                 requirement_groups=groups, max_steps=int(q.get('max_steps', 12)),
                 sufficiency_threshold=float(q.get('sufficiency_threshold', 0.99)), note=q.get('note', ''))
    parsed = pipe.parser.parse(Q.question, kg)
    tmpl = getattr(getattr(parsed, 'query_intent', None), 'template_id', None)
    spec = getattr(parsed, 'evidence_spec', None)
    checks = [c.check_id for c in spec.checks] if spec else None
    rw = type(getattr(parsed, 'rewriter', None)).__name__
    print(q['qid'], '| qtype=', Q.qtype, '| template_id=', tmpl, '| rewriter=', rw)
    print('   spec.checks=', checks)
