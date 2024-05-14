"""
Created on 09/05/2024
@author: Antonin Berthon

Qualitative Similarity extension for a TTS model.
"""
from typing import Tuple, List, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer

from timeview.basis import BSplineBasis

Composition = Tuple[List[int], List[float]]


class TTS_QS(nn.Module):
    def __init__(
        self,
        base_model,
        composition: Composition,
        similarity: Callable[[Composition, Composition], float],
        column_transformer: Optional[ColumnTransformer] = None,
        **similarity_kwargs,
    ):
        """
        Args:
            base_model: an instance of the existing model that has a method `predict_latent_variables`
        """
        super().__init__()
        self.base_model = base_model
        self.bspline = BSplineBasis(
            base_model.config.n_basis,
            (0, base_model.config.T),
            internal_knots=base_model.config.internal_knots,
        )
        self.composition = composition
        self.similarity = similarity
        self.column_transformer = column_transformer
        self.similarity_kwargs = similarity_kwargs

    def forward(self, X):
        """
        Args:
            X: n x m input tensor where n is the number of samples and m is the number of features
        """
        if self.column_transformer is not None:
            X = self.column_transformer.transform(X)  # transform input

        # Get bspline coefficients
        coeffs = self.base_model.predict_latent_variables(X)

        # Get compositions
        current_compositions = []
        for c in coeffs:
            template, transition_points = self.bspline.get_template_from_coeffs(c)
            current_compositions.append((template, transition_points))

        # Compute distance to target composition
        output = np.array(
            [
                self.similarity(
                    s1=self.composition[0],
                    t1=self.composition[1],
                    s2=template,
                    t2=transition_points,
                    **self.similarity_kwargs,
                )
                for template, transition_points in current_compositions
            ]
        )
        return torch.from_numpy(output)
