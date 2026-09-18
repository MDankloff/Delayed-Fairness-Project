"""Activity-aware extension of the original LCF training objective.

No retention penalty is included. Per-round terms use active rows; the long-term
term uses a supplied direct intervention rollout. Legacy callers without a
rollout retain the complete-history conditional-ratio estimator.
"""
import numpy as np
from fair_model import FairModel, to_tensor
from evaluation import compute_post_long_cond_probs


class ActivityAwareFairModel(FairModel):
    name = 'LCF (activity-aware)'

    def train(self, s, OXs, OYs, Xs, Ys, epochs=0, plot=True, tol=1e-7, short_type='pos', OAs=None, As=None,
              long_rollout=None):
        
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
        from direct_fairness import direct_long_term_loss
        if long_rollout is not None:
            if len(long_rollout.Xs) != len(Xs) or not np.array_equal(long_rollout.s, s):
                raise ValueError('Intervention must match the training cohort and horizon.')
        long_probs = None
        if long_rollout is None:
            # Compatibility path for older notebooks only.
            long_mask = np.logical_and.reduce(current_masks)
            long_s = s[long_mask]
            long_Xs = [np.asarray(x)[long_mask] for x in Xs]
            long_Ys = [np.asarray(y)[long_mask] for y in Ys]
            long_probs = compute_post_long_cond_probs(
                long_s, long_Xs, long_Ys, clip=self.probability_clip)
        losses, o_losses, s_fairs, l_fairs = [], [], [], []

        # Mask and convert fixed inputs once per inner optimization call.
        utility_batches = [
            (s[mask], to_tensor(np.asarray(OX)[mask]), to_tensor(np.asarray(Oy)[mask]))
            for OX, Oy, mask in zip(OXs, OYs, original_masks) if mask.any()
        ]
        short_batches = [
            (s[mask], to_tensor(np.asarray(X)[mask]))
            for X, mask in zip(Xs, current_masks) if np.any(s[mask] == 0)
        ]
        gap = 1e30
        pre_loss = 1e30
        while gap > tol or epochs > 0:

            zero = self.linear.weight.sum() * 0.0
            o_loss, s_fair = zero, zero
            for group, OX, Oy in utility_batches:
                o_loss = o_loss + self.compute_loss(group, OX, Oy)
            for group, X in short_batches:
                s_fair_pos, s_fair_neg = self.compute_short_fairness_from_cond_dist(group, X)
                if short_type == 'pos':
                    s_fair = s_fair + s_fair_pos
                if short_type == 'neg':
                    s_fair = s_fair + s_fair_neg

            if long_rollout is not None:
                l_fair = direct_long_term_loss(self, long_rollout)
            else:
                l_fair = self.compute_post_long_fairness_from_cond_dist(
                    long_s, long_Xs, long_Ys, long_probs)

            loss = o_loss + self.sf_reg * s_fair + self.lf_reg * l_fair

            losses.append(loss.item())
            o_losses.append(o_loss.item())
            s_fairs.append(s_fair.item())
            l_fairs.append(l_fair.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            gap = pre_loss - loss
            pre_loss = loss
            epochs -= 1

        if losses:
            self.last_training_summary = dict(
                long_term_estimator=('direct_reference_policy_initial_cohort'
                                     if long_rollout is not None else 'legacy_conditional_ratio'),
                utility_loss=o_losses[-1], short_term_loss=s_fairs[-1],
                long_term_loss=l_fairs[-1], total_loss=losses[-1], inner_steps=len(losses))
        self.save_params()
        if plot:
            self.plot_data(losses, o_losses, s_fairs, l_fairs)
