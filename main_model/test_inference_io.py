"""Standard-library checks; synthetic records only, no model execution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from check_inference_io import check


class InferenceIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root/'input.jsonl'
        self.input.write_text('{"sample_id":"0002","text":"例子,一"}\n'
                              '{"sample_id":"0001","text":"example two"}\n', encoding='utf-8-sig')
        self.pred = self.root/'predictions.csv'
        self.pred.write_text('sample_id,p_machine\n0002,0\n0001,1\n', encoding='utf-8-sig')

    def test_input_and_output_preserve_leading_zeros(self):
        self.assertEqual(check(self.input)['status'], 'PASS_INPUT_FORMAT')
        result = check(self.input, self.pred)
        self.assertEqual(result['status'], 'PASS_INPUT_OUTPUT_FORMAT')
        self.assertEqual(result['rows'], 2)

    def test_csv_quoted_multiline_and_extra_input_fields(self):
        source = self.root/'input.csv'
        source.write_text('sample_id,text,label\n0002,"例子,\n一",ignored\n0001,two,ignored\n', encoding='utf-8')
        self.assertEqual(check(source, self.pred)['rows'], 2)

    def test_bad_json_input(self):
        bad = ['', '[]', '{', '{"sample_id":"a","sample_id":"b","text":"x"}',
               '{"sample_id":1,"text":"x"}', '{"sample_id":true,"text":"x"}',
               '{"sample_id":" ","text":"x"}', '{"sample_id":"a","text":null}',
               '{"sample_id":"a","text":" "}', '{"sample_id":"a"}',
               '{"sample_id":"a","text":"x"}\n{"sample_id":"a","text":"y"}']
        for text in bad:
            with self.subTest(text=text):
                self.input.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    check(self.input)

    def test_bad_csv_structure(self):
        source = self.root/'input.csv'
        for text in ['sample_id,text,text\na,x,y\n', 'sample_id,text\na,x,extra\n',
                     'sample_id,text\na\n', 'sample_id,text\n', ',text\na,x\n']:
            with self.subTest(text=text):
                source.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    check(source)

    def test_prediction_alignment(self):
        for body in ['0001,0\n0002,1\n', '0002,0\n', '0002,0\n0001,1\n0003,.5\n',
                     '0002,0\n0002,1\n', '2,0\n1,1\n', '']:
            with self.subTest(body=body):
                self.pred.write_text('sample_id,p_machine\n'+body, encoding='utf-8')
                with self.assertRaises(ValueError):
                    check(self.input, self.pred)

    def test_invalid_probabilities(self):
        for value in ['NaN', 'Infinity', '-Infinity', '-.1', '1.1', '', 'true']:
            with self.subTest(value=value):
                self.pred.write_text(f'sample_id,p_machine\n0002,{value}\n0001,.5\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    check(self.input, self.pred)

    def test_prediction_columns_and_width(self):
        for text in ['p_machine,sample_id\n0,0002\n1,0001\n',
                     'sample_id,p_machine,extra\n0002,0,x\n0001,1,x\n',
                     'sample_id,p_machine\n0002,0,x\n0001,1\n',
                     'sample_id,p_machine\n0002\n0001,1\n']:
            with self.subTest(text=text):
                self.pred.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    check(self.input, self.pred)

    def test_cli_report_no_text_and_no_overwrite(self):
        report = self.root/'check.json'
        command = [sys.executable, str(Path(__file__).with_name('check_inference_io.py')),
                   '--input', str(self.input), '--predictions', str(self.pred), '--output', str(report)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        original = report.read_bytes()
        self.assertEqual(json.loads(original)['status'], 'PASS_INPUT_OUTPUT_FORMAT')
        self.assertNotIn('example two', original.decode())
        self.assertNotIn('0002', original.decode())
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(report.read_bytes(), original)

    def test_failed_cli_writes_no_pass(self):
        report = self.root/'check.json'
        self.pred.write_text('sample_id,p_machine\n0002,not-a-number\n0001,.5\n', encoding='utf-8')
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('check_inference_io.py')),
                                 '--input', str(self.input), '--predictions', str(self.pred),
                                 '--output', str(report)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('not-a-number', result.stderr)
        self.assertFalse(report.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
