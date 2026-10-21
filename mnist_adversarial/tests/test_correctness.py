import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'attacks'))
sys.path.insert(0, str(ROOT))

from data import DataConfig, get_datasets, get_transform
from evaluate import evaluate_model
from evaluate_attacks import evaluate_attack
from evaluate_attacks import evaluate_suite
from fgsm import fgsm_attack
from pgd import pgd_attack
from sanity import check_adversarial


class AttackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(4, 10))
        self.images = torch.tensor([[[[0., .2], [.7, 1.]]], [[[.8, .1], [.5, .9]]]])
        self.labels = torch.tensor([2, 7])
        self.criterion = nn.CrossEntropyLoss()

    def test_fgsm_matches_input_gradient(self):
        x = self.images.clone().requires_grad_(True)
        grad, = torch.autograd.grad(self.criterion(self.model(x), self.labels), x)
        expected = (x + .3 * grad.sign()).clamp(0, 1).detach()
        actual = fgsm_attack(self.model, self.images, self.labels, self.criterion, .3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(actual.requires_grad)

    def test_pgd_fixed_original_and_fresh_gradients(self):
        original = self.images.clone()
        expected = original.clone()
        for _ in range(8):
            x = expected.detach().requires_grad_(True)
            grad, = torch.autograd.grad(self.criterion(self.model(x), self.labels), x)
            expected = (original + (x + .09 * grad.sign() - original).clamp(-.2, .2)).clamp(0, 1).detach()
        actual = pgd_attack(self.model, original, self.labels, self.criterion, .2, .09, 8, False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        check_adversarial(original, actual, .2)

    def test_random_start_and_zero_epsilon(self):
        torch.manual_seed(17)
        expected = (self.images + torch.empty_like(self.images).uniform_(-.3, .3)).clamp(0, 1)
        torch.manual_seed(17)
        actual = pgd_attack(self.model, self.images, self.labels, self.criterion, steps=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for attack in (fgsm_attack, pgd_attack):
            actual = attack(self.model, self.images, self.labels, self.criterion, epsilon=0)
            torch.testing.assert_close(actual, self.images, rtol=0, atol=0)

    def test_attacks_preserve_inputs_parameter_grads_and_mode(self):
        for attack in (fgsm_attack, pgd_attack):
            for training in (False, True):
                self.model.train(training)
                original, labels = self.images.clone(), self.labels.clone()
                for p in self.model.parameters():
                    p.grad = torch.ones_like(p)
                seen_modes = []
                hook = self.model.register_forward_pre_hook(lambda m, args: seen_modes.append(m.training))
                with torch.no_grad():
                    adv = attack(self.model, self.images, self.labels, self.criterion, epsilon=.3)
                hook.remove()
                self.assertTrue(seen_modes)
                self.assertFalse(any(seen_modes))
                self.assertEqual(self.model.training, training)
                self.assertFalse(adv.requires_grad)
                torch.testing.assert_close(self.images, original)
                torch.testing.assert_close(self.labels, labels)
                for p in self.model.parameters():
                    torch.testing.assert_close(p.grad, torch.ones_like(p))

    def test_training_update_has_only_combined_batch_gradients(self):
        for attack in (fgsm_attack, pgd_attack):
            self.model.train()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
            optimizer.zero_grad(set_to_none=True)
            adv = attack(self.model, self.images, self.labels, self.criterion, epsilon=.3)
            self.assertTrue(all(p.grad is None for p in self.model.parameters()))
            batch = torch.cat([self.images, adv.detach()])
            targets = torch.cat([self.labels, self.labels])
            loss = self.criterion(self.model(batch), targets)
            expected = torch.autograd.grad(loss, tuple(self.model.parameters()))
            loss = self.criterion(self.model(batch), targets)
            loss.backward()
            for p, grad in zip(self.model.parameters(), expected):
                torch.testing.assert_close(p.grad, grad)
            optimizer.step()

    def test_sanity_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            check_adversarial(torch.tensor([.5]), torch.tensor([.8]), .1)
        with self.assertRaises(ValueError):
            check_adversarial(torch.tensor([.9]), torch.tensor([1.1]), .3)
        with self.assertRaises(ValueError):
            fgsm_attack(self.model, self.images + 1, self.labels, self.criterion, .3)

    def test_mode_restored_when_attack_fails(self):
        def failing_loss(logits, labels):
            raise RuntimeError('test failure')

        for attack in (fgsm_attack, pgd_attack):
            self.model.train()
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                attack(self.model, self.images, self.labels, failing_loss, epsilon=.3)
            self.assertTrue(self.model.training)


class TrainingScriptTests(unittest.TestCase):
    def test_real_defense_loops_use_current_model_and_combined_batches(self):
        from defenses import adversarial_training, pgd_adversarial_training

        images = torch.tensor([[[[.2, .3], [.7, .9]]], [[[.8, .1], [.2, .4]]]])
        labels = torch.tensor([2, 7])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        for module, attack_name in ((adversarial_training, 'fgsm_attack'),
                                    (pgd_adversarial_training, 'pgd_attack')):
            model = nn.Sequential(nn.Flatten(), nn.Linear(4, 10))
            original_attack = getattr(module, attack_name)
            previous_weights = []
            training_batches = []

            def attack(m, x, y, criterion, **kwargs):
                self.assertIs(m, model)
                self.assertTrue(m.training)
                self.assertTrue(all(p.grad is None for p in m.parameters()))
                weights = next(m.parameters()).detach().clone()
                if previous_weights:
                    self.assertFalse(torch.equal(weights, previous_weights[-1]))
                previous_weights.append(weights)
                adv = original_attack(m, x, y, criterion, **kwargs)
                self.assertFalse(adv.requires_grad)
                self.assertTrue(m.training)
                self.assertTrue(all(p.grad is None for p in m.parameters()))
                training_batches.append(torch.cat([x, adv]))
                return adv

            def check_batch(m, args):
                if m.training:
                    self.assertFalse(args[0].requires_grad)
                    torch.testing.assert_close(args[0], training_batches[-1])

            hook = model.register_forward_pre_hook(check_batch)
            with (patch.object(module, 'MNISTCNN', return_value=model),
                  patch.object(module, 'get_dataloaders', return_value=(loader, loader, loader)),
                  patch.object(module, 'choose_device', return_value=torch.device('cpu')),
                  patch.object(module, attack_name, side_effect=attack),
                  patch.object(torch, 'save') as save,
                  patch.object(Path, 'mkdir'), patch.object(Path, 'write_text'),
                  contextlib.redirect_stdout(io.StringIO())):
                module.main()
            hook.remove()
            self.assertEqual(len(previous_weights), 5)
            self.assertEqual(save.call_count, 1)


class CheckpointCLITests(unittest.TestCase):
    def test_explicit_checkpoint_and_default(self):
        import evaluate_defended

        for argv, default, expected in (
            (['checkpoints/pgd_adversarially_trained_model.pt'], None,
             Path('checkpoints/pgd_adversarially_trained_model.pt')),
            ([], ROOT / 'checkpoints/pgd_adversarially_trained_model.pt',
             ROOT / 'checkpoints/pgd_adversarially_trained_model.pt'),
        ):
            with (patch.object(evaluate_defended, 'get_dataloaders', return_value=([], [], [])),
                  patch.object(evaluate_defended, 'load_model') as load,
                  patch.object(evaluate_defended, 'evaluate_clean'),
                  patch.object(evaluate_defended, 'evaluate_suite'),
                  contextlib.redirect_stdout(io.StringIO())):
                evaluate_defended.main(argv, default_checkpoint=default)
            self.assertEqual(load.call_args.args[0], expected)


class ThresholdModel(nn.Module):
    def forward(self, x):
        value = x.flatten(1)[:, 0]
        return torch.stack([1 - value, value] + [torch.full_like(value, -10)] * 8, dim=1)


class EvaluationTests(unittest.TestCase):
    def test_confusion_matrix_rows_are_targets(self):
        images = torch.tensor([.1, .1, .9]).reshape(3, 1, 1, 1)
        labels = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = ThresholdModel()
        clean = evaluate_model(model, loader, torch.device('cpu'))
        self.assertEqual(clean.confusion_matrix[:2, :2].tolist(), [[1, 0], [1, 1]])
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_attack(model, loader, torch.device('cpu'), 'identity',
                                     lambda m, x, y, c: x)
        self.assertEqual(result['confusion_matrix'][:2, :2].tolist(), [[1, 0], [1, 1]])
        self.assertEqual(result['attack_success_rate'], 0.)

    def test_empty_loaders_are_rejected(self):
        for evaluate in (evaluate_model,):
            with self.assertRaisesRegex(ValueError, 'empty'):
                evaluate(ThresholdModel(), [], torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'empty'):
            evaluate_attack(ThresholdModel(), [], torch.device('cpu'), 'empty',
                            lambda m, x, y, c: x)

    def test_baseline_and_defense_use_identical_main_attacks(self):
        images = torch.tensor([.2, .8]).reshape(2, 1, 1, 1)
        labels = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = ThresholdModel()
        captures = []

        def capture(m, test_loader, device, name=None, attack_fn=None, **kwargs):
            self.assertIs(test_loader, loader)
            x, y = next(iter(test_loader))
            captures.append((kwargs['epsilon'], attack_fn(m, x, y, nn.CrossEntropyLoss())))
            return {}

        with patch('evaluate_attacks.evaluate_attack', side_effect=capture):
            evaluate_suite(model, loader, torch.device('cpu'))
            baseline = captures[:]
            captures.clear()
            evaluate_suite(model, loader, torch.device('cpu'), defended=True)
        self.assertEqual([eps for eps, _ in baseline[3:]], [.05, .1, .2, .3])
        self.assertEqual([eps for eps, _ in captures], [.1, .2, .3, .3])
        for (_, base), (_, defended) in zip(baseline[:3] + baseline[-1:], captures):
            torch.testing.assert_close(base, defended, rtol=0, atol=0)

    def test_accuracy_asr_confusion_and_weighted_loss(self):
        images = torch.tensor([.1, .9, .2, .8]).reshape(4, 1, 1, 1)
        labels = torch.tensor([0, 1, 1, 0])
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        model = ThresholdModel()
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_attack(model, loader, torch.device('cpu'), 'test', lambda m, x, y, c: 1 - x)
        self.assertEqual(result['clean_accuracy'], .5)
        self.assertEqual(result['adversarial_accuracy'], .5)
        self.assertEqual(result['attack_success_rate'], 1.)
        self.assertEqual(result['total_examples'], 4)
        self.assertEqual(result['confusion_matrix'][:2, :2].tolist(), [[1, 1], [1, 1]])
        expected_loss = nn.CrossEntropyLoss()(model(1 - images), labels).item()
        self.assertAlmostEqual(result['adversarial_loss'], expected_loss, places=6)
        clean = evaluate_model(model, loader, torch.device('cpu'))
        self.assertEqual(clean.accuracy, .5)
        self.assertAlmostEqual(clean.loss, nn.CrossEntropyLoss()(model(images), labels).item(), places=6)

    def test_split_and_pixel_space(self):
        config = DataConfig(data_dir=str(ROOT / 'data'))
        train, val, test = get_datasets(config)
        self.assertEqual((len(train), len(val), len(test)), (55000, 5000, 10000))
        self.assertIs(train.dataset, val.dataset)
        self.assertIsNot(train.dataset, test)
        self.assertFalse(set(train.indices) & set(val.indices))
        expected = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).tolist()
        self.assertEqual(train.indices, expected[:55000])
        self.assertEqual(val.indices, expected[55000:])
        self.assertTrue(train.dataset.train)
        self.assertFalse(test.train)
        image, _ = test[0]
        self.assertEqual(image.shape, (1, 28, 28))
        self.assertGreaterEqual(image.min().item(), 0)
        self.assertLessEqual(image.max().item(), 1)
        self.assertEqual(type(get_transform()).__name__, 'ToTensor')


if __name__ == '__main__':
    unittest.main()
