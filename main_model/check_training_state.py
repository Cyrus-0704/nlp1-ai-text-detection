"""Check cursor continuation and the batch=1 class-weight gradient contract."""
import torch
from training_engine import CursorBatches, classification_loss

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
    print('PASS: resumed cursors cross epoch boundaries exactly; class weights scale batch=1 gradients.')

if __name__ == '__main__':
    main()
