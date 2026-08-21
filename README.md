# MirrorMate – Chess Move Predictor Trained on My Own Games

**MirrorMate** is my summer semester Deep Learning project.  
I trained a convolutional neural network on **my own Chess.com games** so the model learns to play (and predict moves) in a style that looks like me.

This is not a generic Stockfish clone or a random chess bot.  
This is a personal mirror of how *I* play chess.

---

## Why I Built This

I’ve been fascinated by chess since childhood.  
In 2023 I finally started taking it seriously and created a Chess.com account. Since then I’ve played thousands of games.

When our professor gave us complete freedom to build any Deep Learning project (and allowed AI assistance), I immediately knew what I wanted to do:

> Train a neural network on **my own games** so I can see whether the model has actually learned my playing style.

I teamed up with two classmates who shared the same energy:

- [Arun Neupane](https://github.com/arunneupane332)
- [Nistal Gigi Thomas](https://github.com/nistalthomas)

We split the work cleanly:
- Data collection & preprocessing
- Neural network architecture + training
- Interactive frontend (the move predictor UI)

We finished the project, gave a strong presentation, and got good internal marks for it.

---

## What MirrorMate Does

1. Downloads my recent games from Chess.com
2. Converts every position into a 12×8×8 tensor (one channel for each piece type + color)
3. Trains a CNN that predicts **from-square** and **to-square**
4. At inference time, only considers **legal moves** and picks the one the network likes most
5. Shows the prediction live on an interactive chessboard inside the notebook

You can freely set up any position (or start from the standard starting position), choose whose turn it is, and click **Predict Next Move**. The model will play what it thinks *I* would play.

---

## Project Structure & Workflow

### Cell 1 – Setup
- Installs `python-chess`, `torch`, `numpy`, etc.
- Mounts Google Drive
- Creates the folder `ChessBot_Project` and sets the path for the saved model (`chess_model.pth`)

### Cell 2 – Download My Games
- Uses the Chess.com public API
- Username is hardcoded as `mohamedthoufic` (my account)
- Downloads the last few months of games (archives)
- Stores everything in `games_data`

### Cell 3 – Convert Games → Training Data
- Parses every PGN
- For every position:
  - Converts the board into a 12×8×8 matrix (`board_to_matrix`)
  - Records the move that was actually played (`from_square` and `to_square`)
- Result: ~140k training positions (X, y_from, y_to)

### Cell 4 – Train the Model (ChessNet)

Architecture:

```python
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_from = nn.Linear(128*8*8, 64)
        self.fc_to   = nn.Linear(128*8*8, 64)

    def forward(self, x):
        f = self.conv(x)
        return self.fc_from(f), self.fc_to(f)
```

- Two separate heads: one for “from” square, one for “to” square
- Loss = CrossEntropy on from-square + CrossEntropy on to-square
- Trained for several epochs (8 in the final run) with Adam
- Model is saved to Google Drive after training

### Cell 5 – Prediction Backend

This cell creates the function that the frontend will call.

```python
def predict_move_callback(fen_string):
    try:
        board = chess.Board(fen_string)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return json.dumps({"success": False, "error": "No legal moves"})

        tensor = torch.from_numpy(board_to_matrix(board)).unsqueeze(0).to(device).float()
        with torch.no_grad():
            out_from, out_to = model(tensor)

        pf = torch.softmax(out_from, dim=1).cpu().numpy()[0]
        pt = torch.softmax(out_to, dim=1).cpu().numpy()[0]

        # Choose the legal move with highest P(from) × P(to)
        best_move = max(legal_moves, key=lambda m: pf[m.from_square] * pt[m.to_square])
        return json.dumps({"success": True, "move": best_move.uci()})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# Register the function so the HTML can call it
output.register_callback('notebook.predict_move', predict_move_callback)
```

Key points:
- Takes a FEN string from the board
- Only considers legal moves (very important)
- Multiplies the probability of the from-square and to-square
- Returns the best move in UCI format (e.g. `e2e4`)

### Cell 6 – Interactive Frontend (UI)

This is the visual part the user actually interacts with.

- Uses **chessboard.js** + jQuery
- Dark theme that looks clean and modern
- Features:
  - Drag & drop pieces freely
  - Spare pieces so you can set up any position
  - White / Black to move selector
  - “Predict Next Move” button
  - Manual move input (type moves like `e2e4` or `g1f3`)
  - Clear board / Reset to starting position buttons
  - Live status messages (“Thinking…”, “✅ e2e4”, etc.)

When you click **Predict Next Move**, the JavaScript does this:

```javascript
var fen = board.fen() + ' ' + $('#turn_modifier').val() + ' KQkq - 0 1';
google.colab.kernel.invokeFunction('notebook.predict_move', [fen], {})
  .then(function(result) {
      var payload = JSON.parse(result.data['text/plain']);
      if (payload.success) {
          board.move(payload.move.substring(0,2) + '-' + payload.move.substring(2,4));
          $('#status').text('✅ ' + payload.move);
      }
  });
```

The board talks to the Python backend through Colab’s `invokeFunction` bridge.

Later cells in the notebook improve this UI (added manual move input, better error handling, retry logic, etc.), but the core idea stays the same.

---

## Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Language           | Python 3                            |
| Chess logic        | python-chess                        |
| Deep Learning      | PyTorch                             |
| Data               | Chess.com Public API                |
| Frontend           | HTML + chessboard.js + jQuery       |
| Environment        | Google Colab + Google Drive         |
| Model saving       | `.pth` file on Drive                |

---

## How to Run (Google Colab)

1. Open the notebook in Google Colab
2. Run the cells **in order**
3. When you reach the training cell, make sure you have enough RAM / GPU (CPU also works, just slower)
4. After training finishes, the interactive board will appear
5. You can now set up positions and ask the model what it thinks I would play

> Note: The model file is saved on Google Drive. If you re-run the notebook later, it will try to load the already-trained weights.

---

## Personal Notes

Because the model was trained purely on my games, I can instantly tell whether a prediction “feels like me” or not.  
That was the whole point of the project - to create a **mirror** of my own playing style rather than a strong engine.

It is still a simple architecture (no residual blocks, no attention, no value head, no search).  
But for a semester project that had to be built, trained, and presented in limited time, it works surprisingly well as a style imitator.

---

## Team

| Name                                                        | Role                                              |
|-------------------------------------------------------------|---------------------------------------------------|
| Mohamed Thoufic                                             | Idea, model training                              |
| [Arun Neupane](https://github.com/arunneupane332)           | Data collection                                   |
| [Nistal Gigi Thomas](https://github.com/nistalthomas)       | Frontend integration and some backend work        |

---

## License

This project is for educational purposes.  
Feel free to fork it, train it on your own games, and make your own MirrorMate.
