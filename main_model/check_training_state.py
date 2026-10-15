"""Check cursor continuation and the batch=1 class-weight gradient contract."""
import torch
import tempfile
from pathlib import Path
from common import write_json
from training_engine import CursorBatches, classification_loss, load_training_state

def main():
    for batch_size in (1,3):
        cursor = CursorBatches(7,batch_size,42)
        iterator = iter(cursor)
        for _ in range(5):
            next(iterator)
        restored = iter(CursorBatches(7,batch_size,42,cursor.state_dict()))
        assert [next(iterator) for _ in range(20)] == [next(restored) for _ in range(20)]
    logits = torch.tensor([[0.1,-0.2]],requires_grad=True)
    label = torch.tensor([0])
    a = classification_loss(logits,label,torch.ones(2))
    grad_a = torch.autograd.grad(a,logits)[0]
    b = classification_loss(logits,label,torch.tensor([2.,.5]))
    grad_b = torch.autograd.grad(b,logits)[0]
    torch.testing.assert_close(grad_b,2*grad_a)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        root=Path(tmp)
        write_json(root/'resume_manifest.json',dict(signature={'seed':42},files={}))
        try:
            load_training_state(root,{'seed':43})
        except ValueError:
            pass
        else:
            raise AssertionError('Changed resume settings accepted')
        (root/'model.safetensors').write_bytes(b'not a valid checkpoint')
        write_json(root/'resume_manifest.json',dict(signature={},files={'model.safetensors':'0'*64}))
        try:
            load_training_state(root,{})
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt checkpoint accepted')
    print('PASS: cross-epoch cursor restoration, weighted batch=1 gradients, incompatible/corrupt resume rejection.')

if __name__ == '__main__':
    main()
