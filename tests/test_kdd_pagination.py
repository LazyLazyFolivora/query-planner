"""Synthetic contract regressions only; no model/provider or business outcomes."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import radar_policy

def request():
    return json.loads((ROOT/'contracts/radar-policy-v2-pagination.example.json').read_text())
def received(action,more=False,cursor=None):
    return {'action_id':action['id'],'kind':action['kind'],'arguments':copy.deepcopy(action['arguments']),
            'status':'completed','new_evidence_ids':[],
            'pagination':{'has_more':more,'next_cursor':cursor,'max_seq':42,'page_number':action['arguments']['page_number']}}

class PaginationTests(unittest.TestCase):
    def test_completed_three_page_chain_retains_unseen_scope_until_exhausted(self):
        req=request();before=copy.deepcopy(req)
        out=radar_policy.decide(req);self.assertEqual(out['action']['id'],'catalog-page-2');self.assertEqual(req,before)
        req['checks'][0].update(status='passed',evidence_ids=['seen-document-1'])
        # Check completion alone cannot close a known truncated search chain.
        self.assertEqual(radar_policy.decide(req)['decision'],'continue')
        second=copy.deepcopy(req['candidates'][0]);req['history'].append(received(second,True,'cursor-page-3'))
        third=copy.deepcopy(second);third['id']='page-3';third['arguments'].update(cursor='cursor-page-3',previous_action_id=second['id'],page_number=3,index=2)
        req['candidates']=[third];req['budget']['remaining']['tool_calls']-=1
        out=radar_policy.decide(req);self.assertEqual(out['action']['id'],'page-3')
        chains=out['state_summary']['incomplete_search_chains'];self.assertEqual(len(chains),1);self.assertEqual(chains[0]['previous_action_id'],second['id'])
        self.assertNotIn('unseen_record_count',chains[0]);self.assertNotIn('remaining_pages',chains[0])
        req['history'].append(received(third));req['candidates']=[]
        out=radar_policy.decide(req);self.assertEqual(out['decision'],'complete');self.assertEqual(out['state_summary']['incomplete_search_chains'],[])

    def test_unconfirmed_page_requires_reconciliation_and_never_unlocks_another(self):
        for status in ['unknown','prepared','sent','unconfirmed']:
            req=request();action=req['candidates'][0];entry=received(action);entry.pop('pagination');entry['status']=status;req['history'].append(entry)
            req['checks'][0].update(status='passed',evidence_ids=['seen-document-1']);req['budget']['remaining']['tool_calls']=0
            out=radar_policy.decide(req);self.assertEqual(out['reason'],'pending_execution_or_reconciliation');self.assertIsNone(out['action'])

    def test_renamed_metadata_does_not_repeat_an_attempted_cursor(self):
        for status in ['completed','failed','not_sent']:
            req=request();entry=received(req['candidates'][0]);entry['status']=status
            if status!='completed':entry.pop('pagination')
            req['history'].append(entry);a=req['candidates'][0];a['id']='different-display-id';a['arguments']['index']=77;a['arguments']['plan']['reason']='display-only change'
            out=radar_policy.decide(req);self.assertIsNone(out['action']);self.assertIn({'id':a['id'],'reason':'already_attempted'},out['state_summary']['excluded_candidates'])

    def test_scope_query_snapshot_and_previous_receipt_are_not_interchangeable(self):
        for mutate in [lambda a:a.update(cursor='invented'),lambda a:a.update(previous_action_id='missing'),lambda a:a.update(max_seq=43),lambda a:a.update(page_number=3),lambda a:a.update(limit=9),lambda a:a.update(source_id='other-source')]:
            req=request();args=req['candidates'][0]['arguments'];mutate(args)
            if args['source_id']!='fixture-documents':args['plan']['source_id']=args['source_id']
            self.assertIsNone(radar_policy.decide(req)['action'])
        for key,value in [('is_demo',False),('mode','public_history_reconstruction'),('as_of','2026-11-01T00:00:00Z')]:
            req=request();args=req['candidates'][0]['arguments'];args[key]=value;args['scope'][key]=value
            self.assertIsNone(radar_policy.decide(req)['action'])
        req=request();args=req['candidates'][0]['arguments'];args.update(catalog_discovery=False,terms=['unrelated']);args['plan']['terms']=['unrelated']
        self.assertIsNone(radar_policy.decide(req)['action'])

    def test_known_more_pages_do_not_bypass_action_or_shared_budget(self):
        req=request();req['checks'][0].update(status='passed',evidence_ids=['seen-document-1']);req['action_limits']['document_search']=1
        out=radar_policy.decide(req);self.assertEqual(out['reason'],'search_chains_incomplete');self.assertEqual(out['state_summary']['excluded_candidates'][0]['reason'],'action_limit_reached')
        self.assertEqual(out['state_summary']['search_attempt_counts_by_phase'],{'initial':1})
        req=request();req['budget']['remaining']['tool_calls']=0
        self.assertEqual(radar_policy.decide(req)['reason'],'budget_exhausted')
        req=request();req['candidates']=[];self.assertEqual(radar_policy.decide(req)['reason'],'search_chains_incomplete')

    def test_malformed_receipts_or_partial_pagination_cannot_create_candidates(self):
        for mutate in [lambda a:a.update(cursor='x',provider='weknora'),lambda a:a.update(query=None),lambda a:a.update(page_number=True),lambda a:a.update(page_number=9),lambda a:a.update(max_seq=True),lambda a:a.update(limit=2.5),lambda a:a.pop('previous_action_id')]:
            req=request();mutate(req['candidates'][0]['arguments'])
            with self.assertRaises(ValueError):radar_policy.decide(req)
        for mutate in [lambda r:r.update(pagination=None),lambda r:r['pagination'].update(has_more=False),lambda r:r['pagination'].update(max_seq=True),lambda r:r['pagination'].update(page_number=2),lambda r:r['pagination'].update(extra='fake'),lambda r:r.update(status='unknown')]:
            req=request();mutate(req['history'][0])
            with self.assertRaises(ValueError):radar_policy.decide(req)

    def test_pure_entrypoint_rejects_missing_fixed_request_scope(self):
        for field in ["is_demo", "mode", "as_of", "direction"]:
            req=request();req["scope"].pop(field)
            with self.assertRaisesRegex(ValueError,"complete fixed request scope"):
                radar_policy.decide(req)
        req=request();req["scope"]={}
        with self.assertRaisesRegex(ValueError,"complete fixed request scope"):
            radar_policy.decide(req)
        req=request();req["scope"]["is_demo"]=1
        with self.assertRaisesRegex(ValueError,"complete fixed request scope"):
            radar_policy.decide(req)

    def test_keyword_pagination_has_same_binding_and_empty_search_stays_forbidden(self):
        req=request()
        for args in [req['history'][0]['arguments'],req['candidates'][0]['arguments']]:
            args['catalog_discovery']=False;args['terms']=['visible','source'];args['plan']['terms']=['visible','source']
        self.assertEqual(radar_policy.decide(req)['decision'],'continue')
        req['candidates'][0]['arguments']['terms']=['source','visible'];req['candidates'][0]['arguments']['plan']['terms']=['source','visible']
        self.assertEqual(radar_policy.decide(req)['decision'],'continue')
        req['candidates'][0]['arguments']['terms']=[];req['candidates'][0]['arguments']['plan']['terms']=[]
        with self.assertRaises(ValueError):radar_policy.decide(req)

    def test_formal_schema_covers_receipts_and_narrow_continuations(self):
        import jsonschema
        schema=json.loads((ROOT/'contracts/radar-policy-v2.schema.json').read_text());jsonschema.Draft202012Validator.check_schema(schema);jsonschema.validate(request(),schema)
        for mutate in [lambda r:r['candidates'][0]['arguments'].update(query=None),lambda r:r['candidates'][0]['arguments'].update(page_number=1),lambda r:r['candidates'][0]['arguments'].update(provider='weknora'),lambda r:r['history'][0]['pagination'].update(has_more=False),lambda r:r['history'][0].update(status='unknown')]:
            req=request();mutate(req)
            with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(req,schema)

if __name__=='__main__':unittest.main()
