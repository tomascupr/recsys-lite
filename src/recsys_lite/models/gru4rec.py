"""GRU4Rec session-based recommendation model using PyTorch."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from recsys_lite.models.base import BaseRecommender


class SessionDataset(Dataset):
    """Dataset for session-based recommendation."""

    def __init__(self, sessions: List[List[int]], n_items: int) -> None:
        """Initialize session dataset.

        Args:
            sessions: List of session sequences
            n_items: Total number of items
        """
        self.sessions = sessions
        self.n_items = n_items

    def __len__(self) -> int:
        """Get number of sessions.

        Returns:
            Number of sessions
        """
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get session by index.

        Args:
            idx: Session index

        Returns:
            Tuple of (inputs, targets)
        """
        session = self.sessions[idx]
        inputs = torch.LongTensor(session[:-1])
        targets = torch.LongTensor(session[1:])
        return inputs, targets


class GRU4RecModel(nn.Module):
    """GRU4Rec model for session-based recommendation."""

    def __init__(
        self,
        n_items: int,
        hidden_size: int = 100,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Initialize GRU4Rec model.

        Args:
            n_items: Number of items
            hidden_size: Size of hidden layers
            n_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(GRU4RecModel, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Embedding layer
        self.embedding = nn.Embedding(n_items, hidden_size)

        # GRU layers
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )

        # Output layer
        self.out = nn.Linear(hidden_size, n_items)

    def forward(
        self, input_seq: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_seq: Input sequence
            hidden: Hidden state

        Returns:
            Tuple of (output, hidden)
        """
        # Get embeddings
        embedded = self.embedding(input_seq)

        # Initialize hidden state if not provided
        if hidden is None:
            batch_size = input_seq.size(0)
            hidden = self.init_hidden(batch_size)

        # GRU output
        output, hidden = self.gru(embedded, hidden)

        # Reshape for output layer
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.out(output)

        # Ensure we return the expected type for mypy
        return output, hidden  # type: ignore

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state.

        Args:
            batch_size: Batch size

        Returns:
            Initial hidden state
        """
        return torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_size,
            device=self.embedding.weight.device,
        )


class GRU4Rec(BaseRecommender):
    """GRU4Rec wrapper class for session-based recommendation."""

    def __init__(
        self,
        n_items: int,
        hidden_size: int = 100,
        n_layers: int = 1,
        dropout: float = 0.1,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        n_epochs: int = 10,
        use_cuda: bool = False,
    ) -> None:
        """Initialize GRU4Rec wrapper.

        Args:
            n_items: Number of items
            hidden_size: Size of hidden layers
            n_layers: Number of GRU layers
            dropout: Dropout probability
            batch_size: Training batch size
            learning_rate: Learning rate for Adam optimizer
            n_epochs: Number of training epochs
            use_cuda: Whether to use CUDA (GPU)
        """
        self.n_items = n_items
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        # Create model
        self.model = GRU4RecModel(
            n_items=n_items,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the model on user-item interaction data.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        # GRU4Rec needs sessions, not a matrix
        # If sessions are provided in kwargs, use those
        sessions = kwargs.get("sessions", [])
        if not sessions:
            # If no sessions provided, try to create simple ones from matrix
            # This is not optimal but provides compatibility with BaseRecommender
            sessions = []
            for user_idx in range(user_item_matrix.shape[0]):
                items = user_item_matrix[user_idx].indices.tolist()
                if items:
                    sessions.append(items)

        # Call the original fit method with sessions
        self._fit_sessions(sessions)

    def _fit_sessions(self, sessions: List[List[int]]) -> Dict[str, List[float]]:
        """Train the model on session data.

        Args:
            sessions: List of session sequences

        Returns:
            Dictionary of training metrics
        """
        # Create dataset and dataloader
        dataset = SessionDataset(sessions, self.n_items)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Training loop
        metrics: Dict[str, List[float]] = {"loss": []}

        for epoch in range(self.n_epochs):
            epoch_loss = 0

            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).view(-1)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, _ = self.model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Record metrics
            avg_loss = epoch_loss / len(dataloader)
            metrics["loss"].append(avg_loss)

            print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        return metrics

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID
            user_items: Sparse user-item interaction matrix
            n_items: Number of recommendations to return
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (item_ids, scores)
        """
        # For GRU4Rec, we need a session, not just a user ID
        # This is a simplified implementation that assumes the session is passed in kwargs
        session = kwargs.get("session", [])
        return self.predict_next_items(session, n_items)

    def predict_next_items(self, session: List[int], k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next items for a session.

        Args:
            session: Current session sequence
            k: Number of items to recommend

        Returns:
            Tuple of (item_ids, scores)
        """
        self.model.eval()

        with torch.no_grad():
            # Convert session to tensor
            session_tensor = torch.LongTensor([session]).to(self.device)

            # Get predictions
            outputs, _ = self.model(session_tensor)

            # Get scores for last item in sequence
            scores = outputs[-1].cpu().numpy()

            # Get top k items
            top_indices = np.argsort(-scores)[:k]
            top_scores = scores[top_indices]

        return top_indices, top_scores

    def save_model(self, path: str) -> None:
        """Save model to file.

        Args:
            path: File path
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "n_items": self.n_items,
                "hidden_size": self.model.hidden_size,
                "n_layers": self.model.n_layers,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """Load model from file.

        Args:
            path: File path
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Update model parameters
        self.n_items = checkpoint["n_items"]

        # Recreate model with correct dimensions
        self.model = GRU4RecModel(
            n_items=self.n_items,
            hidden_size=checkpoint["hidden_size"],
            n_layers=checkpoint["n_layers"],
        ).to(self.device)

        # Load state dictionaries
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Set to evaluation mode
        self.model.eval()
