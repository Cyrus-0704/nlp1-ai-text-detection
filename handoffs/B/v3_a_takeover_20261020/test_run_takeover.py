"""CPU-only checks for deferred takeover; never launch an experiment."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('run_takeover', HERE / 'run_takeover.py')
takeover = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(takeover)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


class TakeoverTests(unittest.TestCase):
    def setUp(self):
        temp_parent = HERE.parents[2] / '.local' / 'takeover-tests'
        temp_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temp_parent)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.serial = 0

    def fixture(self, executor='B'):
        self.serial += 1
        source = self.base / ('source-' + str(self.serial))
        roots = {}
        for arm in ('T0', 'T1', 'T2'):
            identity = f'{takeover.VERSION}-{executor}-s{takeover.SEEDS[executor]}-{arm}'
            root = source / 'runs' / 'v3' / identity / 'formal'
            roots[arm] = root
            budget = 46800 if executor == 'B' and arm == 'T1' else 39600 if executor == 'B' and arm == 'T0' else 36000
            signature = dict(owner=executor, experiment_id=identity, phase='formal',
                             settings=dict(seed=takeover.SEEDS[executor], arm=arm, max_steps=7500))
            write(root / 'protocol.json', dict(signature=signature, limits=dict(budget_seconds=budget),
                                               evaluation_limits=dict(budget_seconds=1800)))
            protocol_sha = takeover.sha(root / 'protocol.json')
            for name, limit in (('ledger.json', budget), ('evaluation_ledger.json', 1800)):
                write(root / name, dict(protocol_sha256=protocol_sha, budget_seconds=limit,
                                       spent_seconds=1, active=None, events=[]))
            checkpoint = root / 'training' / 'attempt-001' / 'checkpoints' / 'step-007500'
            write(checkpoint / 'resume_manifest.json', dict(signature=signature, optimizer_step=7500))
            training = dict(optimizer_updates=7500, completed=True, checkpoint=str(checkpoint))
            write(root / 'training' / 'attempt-001' / 'training_summary.json',
                  dict(training, status='TRAINED_PENDING_STATE_AND_RELOAD_CHECKS'))
            write(root / 'training' / 'attempt-001' / 'result.json', dict(training, state_verified=True))
            write(root / 'reload_acceptance.json', dict(status='PASS_LOCAL_ACTUAL_MODEL_RELOAD',
                  actual_model_reload_verified=True, reference_process_id=101, reload_process_id=102))
            (root / 'dev_predictions.csv').write_text('sample_id,probability\n1,0.5\n', encoding='utf-8')
            execution = dict(experiment_id=identity, owner=executor, seed=takeover.SEEDS[executor], arm=arm,
                             status='COMPLETE_LOCAL_DEV_NOT_INDEPENDENT_EFFECT', protocol_sha256=protocol_sha,
                             checkpoint=str(checkpoint), training_ledger_sha256=takeover.sha(root / 'ledger.json'),
                             reload_acceptance_sha256=takeover.sha(root / 'reload_acceptance.json'),
                             dev_receipt=dict(actual_model_reload_verified=True,
                                  input_sha256=takeover.DEV_SHA, rows=15086, max_length=512, precision='fp32',
                                  batch_size=1, ordered_ids=True, finite_probability_range=True,
                                  predictions_sha256=takeover.sha(root / 'dev_predictions.csv')),
                             dev_prediction_sha256=takeover.sha(root / 'dev_predictions.csv'))
            write(root / 'execution.json', execution)
        return source, roots

    def test_default_plan_has_expected_queue_and_never_executes(self):
        for executor, arms in (('B', ['T1', 'T2']), ('C', ['T0'])):
            with self.subTest(executor=executor), patch.object(takeover, 'execute') as execute, \
                    patch.object(takeover.subprocess, 'run') as run, \
                    patch.object(takeover, 'completed') as completed, redirect_stdout(io.StringIO()) as output:
                takeover.main(['--executor', executor])
                result = json.loads(output.getvalue())
                self.assertEqual(result['status'], 'DEFERRED_NOT_STARTED')
                self.assertEqual(result['queue'], arms)
                self.assertEqual(result['logical_owner'], 'A')
                self.assertEqual(result['seed'], 43)
                self.assertEqual(result['formal_budget_hours'], 10)
                execute.assert_not_called()
                run.assert_not_called()
                completed.assert_not_called()

    def test_execute_requires_all_three_paths(self):
        options = ('--source-workspace', '--workspace', '--code-archive')
        for missing in (None, *options):
            argv = ['--executor', 'B', '--execute']
            for option in options:
                if missing is not None and option != missing:
                    argv.extend([option, str(self.base / option[2:])])
            with self.subTest(missing=missing), patch.object(takeover, 'execute') as execute, \
                    patch.object(takeover.subprocess, 'run') as run, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    takeover.main(argv)
                self.assertEqual(error.exception.code, 2)
                execute.assert_not_called()
                run.assert_not_called()

    def test_all_three_complete_original_tasks_pass_for_both_executors(self):
        for executor in ('B', 'C'):
            with self.subTest(executor=executor):
                source, roots = self.fixture(executor)
                receipts = takeover.completed(source, executor)
                self.assertEqual(len(receipts), 3)
                for root in roots.values():
                    identity = root.parent.name
                    self.assertEqual(receipts[identity]['execution_sha256'], takeover.sha(root / 'execution.json'))
                    self.assertEqual(receipts[identity]['protocol_sha256'], takeover.sha(root / 'protocol.json'))

    def test_missing_evidence_or_running_original_task_blocks(self):
        for filename in ('execution.json', 'protocol.json', 'ledger.json', 'evaluation_ledger.json',
                         'reload_acceptance.json', 'training/attempt-001/result.json',
                         'training/attempt-001/checkpoints/step-007500/resume_manifest.json'):
            with self.subTest(missing=filename):
                source, roots = self.fixture()
                (roots['T1'] / filename).unlink()
                with self.assertRaises((ValueError, OSError, KeyError)):
                    takeover.completed(source, 'B')
        source, roots = self.fixture()
        path = roots['T2'] / 'execution.json'
        execution = takeover.read(path)
        execution['status'] = 'RUNNING'
        write(path, execution)
        with self.assertRaisesRegex(ValueError, 'not fully complete'):
            takeover.completed(source, 'B')

    def test_7500_without_state_verification_blocks(self):
        for value in (None, False):
            with self.subTest(state_verified=value):
                source, roots = self.fixture()
                path = roots['T0'] / 'training' / 'attempt-001' / 'result.json'
                result = takeover.read(path)
                if value is None:
                    del result['state_verified']
                else:
                    result['state_verified'] = value
                write(path, result)
                with self.assertRaisesRegex(ValueError, 'state verification missing'):
                    takeover.completed(source, 'B')

    def test_active_or_over_budget_ledger_blocks(self):
        for filename in ('ledger.json', 'evaluation_ledger.json'):
            for problem in ('active', 'over_budget'):
                with self.subTest(ledger=filename, problem=problem):
                    source, roots = self.fixture()
                    path = roots['T0'] / filename
                    ledger = takeover.read(path)
                    if problem == 'active':
                        ledger['active'] = dict(owner_pid=999, child_pid=1000)
                    else:
                        ledger['spent_seconds'] = ledger['budget_seconds'] + 1
                    write(path, ledger)
                    execution_path = roots['T0'] / 'execution.json'
                    execution = takeover.read(execution_path)
                    execution['training_ledger_sha256'] = takeover.sha(roots['T0'] / 'ledger.json')
                    write(execution_path, execution)
                    with self.assertRaisesRegex(ValueError, 'unfinished/over-budget'):
                        takeover.completed(source, 'B')

    def test_same_process_reload_blocks_even_with_matching_sha(self):
        source, roots = self.fixture()
        path = roots['T2'] / 'reload_acceptance.json'
        reload = takeover.read(path)
        reload['reload_process_id'] = reload['reference_process_id']
        write(path, reload)
        execution_path = roots['T2'] / 'execution.json'
        execution = takeover.read(execution_path)
        execution['reload_acceptance_sha256'] = takeover.sha(path)
        write(execution_path, execution)
        with self.assertRaisesRegex(ValueError, 'independent reload evidence missing'):
            takeover.completed(source, 'B')

    def test_changed_dev_prediction_blocks(self):
        source, roots = self.fixture()
        (roots['T0'] / 'dev_predictions.csv').write_text('sample_id,probability\n1,0.6\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'dev completion/probability SHA mismatch'):
            takeover.completed(source, 'B')

    def test_dev_receipt_wrong_prediction_or_input_sha_blocks(self):
        for field in ('predictions_sha256', 'input_sha256'):
            with self.subTest(field=field):
                source, roots = self.fixture()
                path = roots['T0'] / 'execution.json'
                execution = takeover.read(path)
                execution['dev_receipt'][field] = '0' * 64
                write(path, execution)
                with self.assertRaisesRegex(ValueError, 'dev completion/probability SHA mismatch'):
                    takeover.completed(source, 'B')

    def test_foreign_checkpoint_blocks_even_with_matching_final_result(self):
        source, roots = self.fixture()
        root = roots['T1']
        foreign = self.base / 'foreign-checkpoint'
        signature = takeover.read(root / 'protocol.json')['signature']
        write(foreign / 'resume_manifest.json', dict(signature=signature, optimizer_step=7500))
        execution_path = root / 'execution.json'
        execution = takeover.read(execution_path)
        execution['checkpoint'] = str(foreign)
        write(execution_path, execution)
        result_path = root / 'training' / 'attempt-001' / 'result.json'
        result = takeover.read(result_path)
        result['checkpoint'] = str(foreign)
        write(result_path, result)
        with self.assertRaisesRegex(ValueError, 'foreign checkpoint'):
            takeover.completed(source, 'B')


if __name__ == '__main__':
    unittest.main()
