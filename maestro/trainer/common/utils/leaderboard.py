from typing import Optional


class CheckpointsLeaderboard:
    def __init__(
        self,
        max_checkpoints: int,
    ) -> None:
        self._max_checkpoints = max(max_checkpoints, 1)
        self._leaderboard: dict[int, tuple[str, float]] = {}

    def register_checkpoint(self, epoch: int, path: str, loss: float) -> tuple[bool, Optional[str]]:
        if len(self._leaderboard) < self._max_checkpoints:
            self._leaderboard[epoch] = (path, loss)
            return True, None
        max_loss_key, max_loss_in_leaderboard = None, None
        for key, (_, loss) in self._leaderboard.items():
            if max_loss_in_leaderboard is None:
                max_loss_key = key
                max_loss_in_leaderboard = loss
            if loss > max_loss_in_leaderboard:  # type: ignore
                max_loss_key = key
                max_loss_in_leaderboard = loss
        if loss >= max_loss_in_leaderboard:  # type: ignore
            return False, None
        to_be_removed, _ = self._leaderboard.pop(max_loss_key)  # type: ignore
        self._leaderboard[epoch] = (path, loss)
        return True, to_be_removed

    def get_best_model(self) -> str:
        min_loss_key, min_loss_in_leaderboard = None, None
        for key, (_, loss) in self._leaderboard.items():
            if min_loss_in_leaderboard is None:
                min_loss_key = key
                min_loss_in_leaderboard = loss
            if loss < min_loss_in_leaderboard:  # type: ignore
                min_loss_key = key
                min_loss_in_leaderboard = loss
        if min_loss_key is None:
            raise RuntimeError("Could not retrieve best model")
        return self._leaderboard[min_loss_key][0]
