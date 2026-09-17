"""Activity-aware extension of the original LCF training objective.

No retention penalty is included. Per-round terms use active rows; the long-term
term uses applicants active throughout the horizon and warns when unestimable.
"""
import warnings
import numpy as np
from fair_model import FairModel, to_tensor
from evaluation import compute_post_long_cond_probs


class ActivityAwareFairModel(FairModel):
    name = 'LCF (activity-aware)'

    def train(self, s, OXs, OYs, Xs, Ys, epochs=0, plot=True, tol=1e-7, short_type='pos', OAs=None, As=None):
        
        # Original trajectories supply prediction loss; current-policy trajectories
        # supply fairness terms. Inactive rows must not enter per-round losses.
        s = np.asarray(s)
        def active_masks(history, features):
            if history is None:
                return [np.ones(len(s), dtype=bool) for _ in features]
            if len(history) != len(features):
                raise ValueError("Activity history must match the feature horizon")
            masks = [np.asarray(a, dtype=bool) for a in history]
            if any(a.shape != s.shape for a in masks):
                raise ValueError("Activity masks must match the applicant population")
            return masks

        original_masks = active_masks(OAs, OXs)
        current_masks = active_masks(As, Xs)
        # A common complete-history cohort is needed by the existing trajectory
        # estimator. With absorbing opt-out these are the final active applicants.
        long_mask = np.logical_and.reduce(current_masks)
        long_s = s[long_mask]
        long_Xs = [np.asarray(x)[long_mask] for x in Xs]
        long_Ys = [np.asarray(y)[long_mask] for y in Ys]
        can_estimate_long = all(np.any(long_s == group) for group in (0, 1))
        if can_estimate_long:
            can_estimate_long = all(
                np.unique(y[long_s == group]).size == 2
                for y in long_Ys[:-1] for group in (0, 1)
            )
        if not can_estimate_long:
            warnings.warn(
                "LCF long-term penalty omitted for this rollout: insufficient "
                "surviving groups or label classes to fit conditional ratios.",
                RuntimeWarning, stacklevel=2,
            )
        long_probs = (compute_post_long_cond_probs(long_s, long_Xs, long_Ys)
                      if can_estimate_long else None)
        losses, o_losses, s_fairs, l_fairs = [], [], [], []

        gap = 1e30
        pre_loss = 1e30
        while gap > tol or epochs > 0:

            zero = self.linear.weight.sum() * 0.0
            o_loss, s_fair = zero, zero
            for OX, Oy, mask in zip(OXs, OYs, original_masks):
                if mask.any():
                    o_loss = o_loss + self.compute_loss(
                        s[mask], np.asarray(OX)[mask], to_tensor(np.asarray(Oy)[mask])
                    )
            for X, mask in zip(Xs, current_masks):
                if np.any(s[mask] == 0):
                    s_fair_pos, s_fair_neg = self.compute_short_fairness_from_cond_dist(
                        s[mask], np.asarray(X)[mask]
                    )
                    if short_type == 'pos':
                        s_fair = s_fair + s_fair_pos
                    if short_type == 'neg':
                        s_fair = s_fair + s_fair_neg

            l_fair = (self.compute_post_long_fairness_from_cond_dist(
                long_s, long_Xs, long_Ys, long_probs
            ) if can_estimate_long else zero)

            loss = o_loss + self.sf_reg * s_fair + self.lf_reg * l_fair

            losses.append(loss.item())
            o_losses.append(o_loss.item())
            s_fairs.append(s_fair.item())
            l_fairs.append(l_fair.item())

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            gap = pre_loss - loss
            pre_loss = loss
            epochs -= 1

        self.save_params()
        if plot:
            self.plot_data(losses, o_losses, s_fairs, l_fairs)

