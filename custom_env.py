from pyquaticus.envs.pyquaticus import PyQuaticusEnv
import pygame


class PyQuaticusCustomEnv(PyQuaticusEnv):

    def _draw_hud(self):
        if self.screen is None:
            return

        font_score = pygame.font.SysFont(None, 24)
        font_ctrls = pygame.font.SysFont(None, 16)

        score = self.game_score

        # Scores
        self.screen.blit(
            font_score.render(
                f"Red Captures: {score['red_captures']}", True, (0, 0, 0)
            ),
            (20, 20),
        )

        self.screen.blit(
            font_score.render(
                f"Blue Captures: {score['blue_captures']}", True, (0, 0, 0)
            ),
            (self.screen.get_width() // 2 + 5, 20),
        )

        # Controls
        y = self.screen.get_height() - 50
        for line in ("SPACE = Pause / Resume", "ESC = Quit"):
            self.screen.blit(
                font_ctrls.render(line, True, (0, 0, 0)),
                (20, y),
            )
            y += 18
