from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import interactive_pcn_zoo_cf as cf
from pcn.env_setup import scaled_command_tensor


def configure_art_local_path():
    repo_root = Path(__file__).resolve().parent
    # ART creates a config directory on import. Keep it local to this repo so
    # the script works in restricted environments and does not depend on HOME.
    os.environ["USERPROFILE"] = str(repo_root)
    os.environ["HOME"] = str(repo_root)
    (repo_root / ".art").mkdir(parents=True, exist_ok=True)


configure_art_local_path()

from art.attacks.evasion import CarliniL2Method  # noqa: E402
from art.estimators.classification import PyTorchClassifier  # noqa: E402


DEFAULT_CW_SETTINGS = {
    "confidence": 0.0,
    "learning_rate": 0.001,
    "binary_search_steps": 12,
    "max_iter": 20000,
    "initial_const": 1.0,
    "max_halving": 5,
    "max_doubling": 5,
    "batch_size": 16,
    "verbose": False,
    "art_clip_relaxation": 1e-2,
    "clip_command": True,
}


@dataclass
class CwSettings:
    confidence: float
    learning_rate: float
    binary_search_steps: int
    max_iter: int
    initial_const: float
    max_halving: int
    max_doubling: int
    batch_size: int
    verbose: bool
    art_clip_relaxation: float
    clip_command: bool


class FixedObservationDeltaModel(torch.nn.Module):
    """Expose command perturbations as ART inputs while keeping the PCN observation fixed."""

    def __init__(self, pcn_model, obs, base_command, action_mask, device):
        super().__init__()
        self.pcn_model = pcn_model
        self.device = device
        obs_tensor = torch.as_tensor(np.asarray(obs)).to(device)
        base_command_tensor = torch.as_tensor(np.asarray(base_command), dtype=torch.float32).to(device)
        self.register_buffer("fixed_obs", obs_tensor)
        self.register_buffer("base_command", base_command_tensor)
        if action_mask is None:
            self.action_mask = None
        else:
            self.register_buffer("action_mask", torch.as_tensor(action_mask, dtype=torch.bool).to(device))

    def forward(self, perturbation_batch):
        perturbation_batch = perturbation_batch.to(self.device, dtype=torch.float32)
        command_batch = self.base_command.unsqueeze(0) + perturbation_batch
        batch_size = int(command_batch.shape[0])
        obs_batch = self.fixed_obs.unsqueeze(0).expand((batch_size,) + tuple(self.fixed_obs.shape))
        outputs = self._pcn_logits(obs_batch, command_batch)
        action_mask = getattr(self, "action_mask", None)
        if action_mask is not None:
            invalid = ~action_mask
            if torch.any(invalid):
                outputs = outputs.clone()
                outputs[:, invalid] = -1.0e9
        return outputs

    def _pcn_logits(self, obs_batch, command_batch):
        fc = getattr(self.pcn_model, "fc", None)
        if (
            isinstance(fc, torch.nn.Sequential)
            and len(fc) > 0
            and isinstance(fc[-1], torch.nn.LogSoftmax)
            and hasattr(self.pcn_model, "s_emb")
            and hasattr(self.pcn_model, "c_emb")
        ):
            command_embedding = self.pcn_model.c_emb(self._scaled_command(command_batch))
            state_embedding = self._state_embedding(obs_batch)
            return fc[:-1](state_embedding * command_embedding)
        return self.pcn_model(obs_batch, command_batch)

    def _scaled_command(self, command_batch):
        if hasattr(self.pcn_model, "n_objectives"):
            return command_batch.float() * self.pcn_model.scaling_factor[:, : self.pcn_model.n_objectives]
        return scaled_command_tensor(command_batch, self.pcn_model.scaling_factor)

    def _state_embedding(self, obs_batch):
        if hasattr(self.pcn_model, "encode_state"):
            return self.pcn_model.s_emb(self.pcn_model.encode_state(obs_batch))
        if hasattr(self.pcn_model, "decode_state") and hasattr(self.pcn_model, "position_embedding"):
            state = obs_batch.float()
            cell_index, inventory = self.pcn_model.decode_state(state)
            position_embedding = self.pcn_model.position_embedding(cell_index)
            inventory_embedding = self.pcn_model.inventory_encoder(inventory)
            return self.pcn_model.s_emb(torch.cat((position_embedding, inventory_embedding), dim=-1))
        return self.pcn_model.s_emb(obs_batch.float())


def choose_cw_settings():
    return CwSettings(**dict(DEFAULT_CW_SETTINGS))


def one_hot_target(action, nb_classes):
    target = np.zeros((1, int(nb_classes)), dtype=np.float32)
    target[0, int(action)] = 1.0
    return target


def art_delta_clip_values(bounds, base_command, relaxation=0.0):
    low, high = bounds
    base_command = np.asarray(base_command, dtype=np.float32)
    relaxation = max(0.0, float(relaxation))
    low = np.asarray(low, dtype=np.float32) - base_command - np.float32(relaxation)
    high = np.asarray(high, dtype=np.float32) - base_command + np.float32(relaxation)
    return low.astype(np.float32), high.astype(np.float32)


def make_art_classifier(model, obs, base_command, action_mask, command_dim, nb_actions, bounds, device, clip_relaxation):
    wrapped = FixedObservationDeltaModel(model, obs, base_command, action_mask, device).to(device)
    wrapped.eval()
    clip_values = art_delta_clip_values(bounds, base_command, relaxation=clip_relaxation)
    device_type = "gpu" if getattr(device, "type", str(device)) == "cuda" else "cpu"
    return PyTorchClassifier(
        model=wrapped,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(int(command_dim),),
        nb_classes=int(nb_actions),
        clip_values=clip_values,
        device_type=device_type,
    )


def run_cw_attack(
    model,
    obs,
    base_command,
    a_foil,
    action_mask,
    settings,
    device,
    bounds,
):
    base_command = np.asarray(base_command, dtype=np.float32)
    classifier = make_art_classifier(
        model,
        obs,
        base_command,
        action_mask,
        command_dim=len(base_command),
        nb_actions=len(action_mask),
        bounds=bounds,
        device=device,
        clip_relaxation=settings.art_clip_relaxation,
    )
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=float(settings.confidence),
        targeted=True,
        learning_rate=float(settings.learning_rate),
        binary_search_steps=int(settings.binary_search_steps),
        max_iter=int(settings.max_iter),
        initial_const=float(settings.initial_const),
        max_halving=int(settings.max_halving),
        max_doubling=int(settings.max_doubling),
        batch_size=int(settings.batch_size),
        verbose=bool(settings.verbose),
    )
    art_low, art_high = classifier.clip_values
    x = np.clip(np.zeros_like(base_command, dtype=np.float32), art_low, art_high).reshape(1, -1).astype(np.float32)
    y = one_hot_target(a_foil, len(action_mask))
    adversarial = attack.generate(x=x, y=y)
    command = base_command + np.asarray(adversarial[0], dtype=np.float32)
    if settings.clip_command:
        command = np.clip(command, bounds[0], bounds[1]).astype(np.float32)
    return command


def eval_settings_from_cw(settings):
    values = dict(cf.DEFAULT_ZOO_SETTINGS)
    values["margin"] = float(settings.confidence)
    values["clip_command"] = bool(settings.clip_command)
    return cf.ZooSettings(**values)


def save_cw_results(
    output_dir,
    env,
    env_name,
    timestep,
    original,
    counterfactual,
    a_star,
    a_foil,
    settings,
):
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "attack": "ART CarliniL2Method",
        "env": env_name,
        "timestep": int(timestep),
        "a_star": int(a_star),
        "a_star_label": cf.action_label(env, a_star),
        "a_foil": int(a_foil),
        "a_foil_label": cf.action_label(env, a_foil),
        "success": bool(counterfactual.success),
        "foil_is_greedy_action": bool(counterfactual.greedy_action == a_foil),
        "l2_norm": float(counterfactual.l2_norm),
        "settings": cf.to_jsonable(settings.__dict__),
        "original": {
            "command": cf.to_jsonable(original.command),
            "greedy_action": int(a_star),
            "greedy_action_label": cf.action_label(env, a_star),
            "probs": cf.to_jsonable(original.probs),
            "target_margin": float(original.target_margin),
        },
        "counterfactual": {
            "command": cf.to_jsonable(counterfactual.command),
            "delta": cf.to_jsonable(counterfactual.delta),
            "greedy_action": int(counterfactual.greedy_action),
            "greedy_action_label": cf.action_label(env, counterfactual.greedy_action),
            "probs": cf.to_jsonable(counterfactual.probs),
            "target_margin": float(counterfactual.target_margin),
        },
    }

    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(cf.to_jsonable(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    cf.plot_action_probs_before_after(
        output_dir / "action_probs_before_after.png",
        env,
        original.probs,
        counterfactual.probs,
    )


def main():
    print("=" * 72)
    print("PCN C&W Command-Space Counterfactual Explainer")
    print("=" * 72)

    context = cf.load_run_and_model()
    run_dir = context["run_dir"]
    setup = context["setup"]
    env = context["env"]
    model = context["model"]
    device = context["device"]

    try:
        desired_return, _ = cf.choose_desired_return(run_dir, setup)
        selection = cf.choose_timestep_with_render(context, desired_return)
        timestep = selection["timestep"]
        rollout = selection["rollout"]
        obs_t = rollout["obs"]
        remaining_return_t = rollout["remaining_return"]
        original_policy = selection["original_policy"]
        action_mask = selection["action_mask"]
        valid_actions = selection["valid_actions"]
        a_star = int(original_policy["action"])

        cf.print_selected_state(
            env,
            desired_return,
            timestep,
            obs_t,
            remaining_return_t,
            valid_actions,
            original_policy,
        )

        foil_actions = [int(action) for action in valid_actions if int(action) != a_star]
        if not foil_actions:
            raise RuntimeError("No valid foil actions are available at this state.")

        print(f"\nOriginal greedy action: {cf.format_action(env, a_star)}")
        while True:
            print("\nFoil action choices:")
            for action in foil_actions:
                print(f"  {cf.format_action(env, action)}")
            a_foil = cf.ask_int("Foil action id", minimum=0, maximum=len(original_policy["masked_log_probs"]) - 1)
            if a_foil not in set(foil_actions):
                print(f"Choose one of the listed foil actions: {cf.format_action_list(env, foil_actions)}")
                continue
            break

        settings = choose_cw_settings()
        eval_settings = eval_settings_from_cw(settings)
        bounds = cf.command_bounds(setup)
        clip_bounds = bounds if settings.clip_command else None

        original_result = cf.evaluate_delta(
            model,
            obs_t,
            remaining_return_t,
            np.zeros_like(remaining_return_t, dtype=np.float32),
            a_star,
            a_foil,
            action_mask,
            eval_settings,
            device,
            clip_bounds=clip_bounds,
        )

        print("\nRunning C&W Reward Perturbation...")
        counterfactual_command = run_cw_attack(
            model,
            obs_t,
            remaining_return_t,
            a_foil,
            action_mask,
            settings,
            device,
            bounds,
        )
        final_result = cf.evaluate_delta(
            model,
            obs_t,
            remaining_return_t,
            counterfactual_command - np.asarray(remaining_return_t, dtype=np.float32),
            a_star,
            a_foil,
            action_mask,
            eval_settings,
            device,
            clip_bounds=clip_bounds,
        )

        cf.print_final_result(env, final_result, a_foil)

        output_dir = (
            Path(run_dir)
            / "counterfactuals"
            / f"cw_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        save_cw_results(
            output_dir,
            env,
            setup.env_name,
            timestep,
            original_result,
            final_result,
            a_star,
            a_foil,
            settings,
        )
        print(f"\nSaved counterfactual artifacts to: {output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
