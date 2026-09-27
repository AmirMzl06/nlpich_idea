"""Small CPU integration tests; no CORP data needed."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from nlp21_jigsaw.cli import train_parser
from nlp21_jigsaw.config import DEFAULT_ARMS
from nlp21_jigsaw.model import JigsawCTC
from nlp21_jigsaw.noise import gaussian_noise
from nlp21_jigsaw.data import TrainingStream
from nlp21_jigsaw.engine import (seed_all, train_step, ctc_loss, make_optimizer,
                                save_checkpoint, load_checkpoint, restore_rng, evaluate)

torch.set_num_threads(1)


def config(arm="proposed", mode="joint", site="neural"):
    args = vars(train_parser().parse_args([
        "--datasetPath", "unused", "--out_dir", "unused", "--jigsaw_arm", arm,
        "--train_mode", mode, "--noise_site", site, "--gru", "--bidir"]))
    args.update(hidden=8, layers=2, dropout=0.1, kernel=8, stride=4,
                nInputFeatures=4, jigsaw_dim=8, jigsaw_width=8, head_width=8,
                ssl_batch=4, noise_std=0.3, seed=3)
    return args


def samples():
    seed_all(31)
    return [(torch.randn(t, 4), "aba", i) for i, t in enumerate([82, 73, 79, 85])]


def new_model(args):
    seed_all(args["seed"])
    model = JigsawCTC(args)
    model.set_quantiles(torch.randn(100, 4))
    return model


class Integration(unittest.TestCase):
    def test_all_16_cells_gradients_and_freeze(self):
        for arm in DEFAULT_ARMS:
            for mode in ("frozen", "joint"):
                for site in ("neural", "embedding"):
                    with self.subTest(arm=arm, mode=mode, site=site):
                        a = config(arm, mode, site); m = new_model(a)
                        stream = TrainingStream(samples(), 2, 19); batch = stream.next()
                        before_encoder = [p.detach().clone() for p in m.encoder.parameters()]
                        stage = "pretrain" if mode == "frozen" else "joint"
                        o = make_optimizer(m, a, stage)
                        metrics = train_step(m, o, batch, a, stage)
                        self.assertTrue(torch.isfinite(torch.tensor(metrics["total"])))
                        self.assertTrue(any(not torch.equal(old, p) for old, p in zip(before_encoder, m.encoder.parameters())))
                        if arm in ("order_lag", "order_time_axis", "order_full"):
                            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.lag_head.parameters()))
                        if mode == "frozen":
                            m.freeze_encoder()
                            frozen = [p.detach().clone() for p in m.ssl_parameters()]
                            decoder_before = [p.detach().clone() for p in m.decoder_parameters()]
                            o = make_optimizer(m, a, "decoder")
                            train_step(m, o, batch, a, "decoder")
                            self.assertTrue(all(torch.equal(old, p) for old, p in zip(frozen, m.ssl_parameters())))
                            self.assertTrue(all(p.grad is None for p in m.ssl_parameters()))
                            self.assertTrue(any(not torch.equal(old, p) for old, p in zip(decoder_before, m.decoder_parameters())))

    def test_gaussian_sigma_mask_and_gradient(self):
        x = torch.zeros(8, 5000, 32, requires_grad=True)
        lengths = torch.tensor([2500] * 8)
        noisy = gaussian_noise(x, lengths, 0.6)
        self.assertLess(abs(float(noisy[:, :2500].detach().std()) - 0.6), 0.003)
        self.assertLess(abs(float(noisy[:, :2500].detach().mean())), 0.003)
        self.assertEqual(float(noisy[:, 2500:].detach().abs().sum()), 0)
        noisy.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))
        self.assertIs(gaussian_noise(x, lengths, 0), x)

    def test_dense_equals_original_window_encoder_and_padding_invariance(self):
        for arm in ("proposed", "order_full"):
            m = new_model(config(arm)); m.eval()
            x = torch.randn(2, 82, 4); lengths = torch.tensor([82, 73])
            prep = m.preprocess(x, lengths)
            z = m.encode_preprocessed(prep, lengths)
            w = m.spec["window"]
            for b, length in enumerate(lengths.tolist()):
                natural = prep[b, :length].T
                padded = torch.nn.functional.pad(natural, (w // 2, w - w // 2 - 1), mode="replicate")
                tiles = padded.unfold(-1, w, 1).permute(1, 0, 2)
                torch.testing.assert_close(z[b, :length], m.encoder(tiles), atol=1e-6, rtol=1e-5)
            solo, sl = m(x[1:2, :73], torch.tensor([73]))
            mixed, ml = m(x, lengths)
            self.assertEqual(int(sl[0]), (73 - m.kernel) // m.stride + 1)
            torch.testing.assert_close(solo[0], mixed[1, :int(ml[1])], atol=2e-6, rtol=1e-5)

    def test_ctc_repeated_characters_guard(self):
        logits = torch.randn(1, 3, 32)
        target = torch.tensor([[6, 6, 7]])
        with self.assertRaisesRegex(ValueError, "impossible"):
            ctc_loss(logits, torch.tensor([3]), target, torch.tensor([3]))

    def test_plan_never_crosses_trials_and_clean_target(self):
        m = new_model(config()); x = torch.randn(2, 82, 4); lens = torch.tensor([82, 70])
        clean = m.preprocess(x, lens); plan = m.sample_plan(clean, lens, 64)
        self.assertTrue((plan["times"][:, -1, -1] < lens[plan["trial"]]).all())
        target = plan["target"].clone()
        m.ssl_loss(clean + 10, plan)
        self.assertTrue(torch.equal(target, plan["target"]))
        self.assertEqual(m.spec["neuron_dropout"], 0)
        self.assertEqual(m.spec["gain_jitter"], 0)

    def test_ctc_alone_reaches_encoder_with_embedding_noise(self):
        m = new_model(config(site="embedding")); batch = TrainingStream(samples(), 2, 3).next()
        x, y, xl, yl, _ = batch
        z = gaussian_noise(m.encode(x, xl), xl, 0.4)
        pred, lengths = m.decode(z, xl)
        ctc_loss(pred, lengths, y, yl).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters()))

    def test_resume_exact_next_update(self):
        a = config(); m = new_model(a); stream = TrainingStream(samples(), 2, 9)
        optimizer = make_optimizer(m, a, "joint")
        train_step(m, optimizer, stream.next(), a, "joint")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, m, optimizer, a, "joint", 1, stream, 1.0)
            train_step(m, optimizer, stream.next(), a, "joint")
            expected = copy.deepcopy(m.state_dict())
            loaded = load_checkpoint(path)
            m2 = new_model(a); m2.load_state_dict(loaded["model"])
            o2 = make_optimizer(m2, a, "joint"); o2.load_state_dict(loaded["optimizer"])
            s2 = TrainingStream(samples(), 2, 9); s2.load_state_dict(loaded["stream"])
            restore_rng(loaded)
            train_step(m2, o2, s2.next(), a, "joint")
            for key, value in expected.items():
                torch.testing.assert_close(value, m2.state_dict()[key], atol=0, rtol=0)

    def test_head_reset_and_cer(self):
        a = config("order_full"); m = new_model(a)
        stream = TrainingStream(samples(), 2, 9); o = make_optimizer(m, a, "joint")
        batch = stream.next(); train_step(m, o, batch, a, "joint")
        param = next(m.order_head.parameters()); old = param.detach().clone()
        self.assertIn(param, o.state)
        m.maybe_reset_heads(400, o)
        self.assertNotIn(param, o.state)
        self.assertFalse(torch.equal(old, param))
        report, rows, _ = evaluate(m, [batch], "cpu")
        self.assertEqual(report["characters"], 6)
        self.assertEqual(report["errors"], sum(r["errors"] for r in rows))
        self.assertAlmostEqual(report["cer"], report["errors"] / 6)


if __name__ == "__main__":
    unittest.main()
