"""Measure real forward/backward agreement and a local multi-agent run."""
import json
import platform
from pathlib import Path
import numpy as np
from .complete_models import (scratch_deep, keras_deep, torch_deep,
                              align_weights, train_batch)


def main():
    import torch
    import keras
    torch.set_num_threads(2)
    from src.tieuluan.scratch.optim import Adam
    from src.tieuluan.scratch.losses import BCEWithLogitsLoss
    out = Path('results/tieuluan/complete')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ['cnn', 'rnn', 'lstm']:
        rng = np.random.default_rng(17)
        shape = (4, 1, 20, 20) if kind == 'cnn' else (4, 20, 1)
        x = rng.normal(size=shape).astype('float32')
        y = np.array([[0], [1], [0], [1]], dtype='float32')
        scratch = scratch_deep(kind)
        pt = torch_deep(kind)
        kr = keras_deep(kind)
        align_weights(scratch, pt, 'pytorch', kind)
        align_weights(scratch, kr, 'keras', kind)
        a = scratch.forward(x)
        b = pt(torch.from_numpy(x)).detach().numpy()
        c = kr(x, training=False).numpy()
        errors = {'pytorch': float(np.max(np.abs(a-b))),
                  'keras': float(np.max(np.abs(a-c)))}
        assert max(errors.values()) < 1e-4, errors
        scratch.zero_grad()
        criterion = BCEWithLogitsLoss()
        criterion.forward(scratch.forward(x), y)
        scratch.backward(criterion.backward())
        pt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            pt(torch.from_numpy(x)), torch.from_numpy(y)).backward()
        sg = [p.grad for p in scratch.parameters()]
        tg = [p.grad.detach().numpy() for p in pt.parameters()
              if p.requires_grad]
        # Scratch carries frozen second biases as implementation buffers.
        sp = [p for p in scratch.parameters()
              if kind == 'cnn' or p.name != 'bias_hh']
        grad_error = max(float(np.max(np.abs(s.grad-t)))
                         for s, t in zip(sp, tg, strict=True))
        assert grad_error < 1e-4, grad_error
        models = {'scratch': scratch, 'pytorch': pt, 'keras': kr}
        optimizers = {'scratch': Adam(scratch.parameters()),
                      'pytorch': torch.optim.Adam(
                          [p for p in pt.parameters() if p.requires_grad],
                          lr=0.001, eps=1e-8),
                      'keras': keras.optimizers.Adam(
                          learning_rate=0.001, epsilon=1e-8)}
        for fw, model in models.items():
            losses = [train_batch(model, fw, x, y, optimizers[fw])
                      for _ in range(3)]
            assert np.isfinite(losses).all()
            row = dict(kind=kind, framework=fw, trainable_layers=6,
                       parameters=sum(p.value.size for p in sp),
                       output_shape=list(a.shape), loss_first=losses[0],
                       loss_third=losses[-1],
                       initial_logit_max_error=errors.get(fw, 0.0),
                       scratch_torch_gradient_max_error=grad_error)
            rows.append(row)
            print(json.dumps(row), flush=True)
    from deployment.chapter04.app import rag
    from src.chapter04.knowledge import load_graph
    from .complete_agents import MultiAgentTeam
    engine = rag()
    graphs = {'finance': load_graph(
        'knowledge_graph/chapter04/finance_graph.json.gz')}
    team = MultiAgentTeam(engine, graphs)
    demonstrations = []
    for domain, question in [
        ('diabetes', 'How many encounters are in the dataset?'),
        ('finance', 'What is the percentage change in revenue?')]:
        state = team.ask(question, domain)
        assert len(state.messages) == 4
        assert state.messages[-1].payload['citation_ids_valid']
        demonstrations.append(dict(
            domain=domain, question=question, answer=state.answer,
            status=state.status,
            evidence_ids=[e['id'] for e in state.evidence],
            graph_matches=len(state.graph_context),
            messages=[vars(m) for m in state.messages]))
    state = team.ask('Question with no matching report', 'finance',
                     report='__missing_report__')
    assert state.status == 'insufficient_evidence'
    checks = []
    for question, domain in [('', 'finance'), ('test', 'invalid')]:
        try:
            team.ask(question, domain)
        except ValueError:
            checks.append('rejected')
        else:
            raise AssertionError('Invalid request accepted')
    report = dict(scope='Synthetic mini-batch verification, not market evaluation',
                  python=platform.python_version(), models=rows,
                  multi_agent=demonstrations,
                  empty_evidence_status=state.status,
                  invalid_request_checks=checks)
    (out/'verification.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('Verified 9 models and 5 multi-agent scenarios', flush=True)


if __name__ == '__main__':
    main()
