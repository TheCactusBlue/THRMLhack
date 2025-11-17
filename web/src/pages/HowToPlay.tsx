import { Link } from "react-router-dom";

export function HowToPlay() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-900 via-neutral-800 to-neutral-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-emerald-400 to-blue-500 bg-clip-text text-transparent">
            How to Play Thermodynamic Tactics
          </h1>
          <Link
            to="/"
            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 rounded-lg font-semibold transition-colors"
          >
            ← Back to Game
          </Link>
        </div>

        {/* Game Overview */}
        <section className="mb-12 bg-neutral-800/50 rounded-xl p-6 border border-neutral-700">
          <h2 className="text-3xl font-bold mb-4 text-emerald-400">
            🎮 Game Overview
          </h2>
          <p className="text-lg text-gray-300 leading-relaxed">
            Thermodynamic Tactics is a turn-based strategy game where two
            players compete by manipulating an Ising model grid through biases
            and couplings. After both players make their moves, a THRML-powered
            Gibbs sampling algorithm determines the final board state. The
            player who controls more territory wins the round. First to win 3 of
            5 rounds wins the game!
          </p>
        </section>

        {/* Ising Model Physics */}
        <section className="mb-12 bg-neutral-800/50 rounded-xl p-6 border border-neutral-700">
          <h2 className="text-3xl font-bold mb-4 text-blue-400">
            ⚛️ The Physics Behind the Game
          </h2>
          <div className="space-y-4 text-gray-300">
            <p className="text-lg leading-relaxed">
              The game is based on the <strong>Ising model</strong>, a
              mathematical model from statistical physics:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li>
                <strong className="text-blue-300">Spins:</strong> Each grid cell
                is a spin that can be +1 (blue/Player A) or -1 (red/Player B)
              </li>
              <li>
                <strong className="text-blue-300">
                  Biases (h<sub>i</sub>):
                </strong>{" "}
                Push individual spins toward +1 or -1
              </li>
              <li>
                <strong className="text-blue-300">
                  Couplings (J<sub>ij</sub>):
                </strong>{" "}
                Control alignment between neighboring spins
                <ul className="list-circle list-inside ml-6 mt-1">
                  <li>Positive coupling → neighbors prefer to be the same</li>
                  <li>Negative coupling → neighbors prefer to be opposite</li>
                </ul>
              </li>
              <li>
                <strong className="text-blue-300">Energy:</strong> E = -Σh
                <sub>i</sub>·s<sub>i</sub> - Σ J<sub>ij</sub>·s<sub>i</sub>·s
                <sub>j</sub> (the system minimizes energy)
              </li>
              <li>
                <strong className="text-blue-300">Beta (β):</strong> Inverse
                temperature (higher = more deterministic, currently 3.0)
              </li>
            </ul>
          </div>
        </section>

        {/* Game Flow */}
        <section className="mb-12 bg-neutral-800/50 rounded-xl p-6 border border-neutral-700">
          <h2 className="text-3xl font-bold mb-4 text-purple-400">
            🔄 Game Flow
          </h2>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <span className="text-3xl font-bold text-purple-300">1.</span>
              <div>
                <h3 className="text-xl font-semibold text-purple-200">
                  Plan Your Strategy
                </h3>
                <p className="text-gray-300">
                  Spend your tokens to adjust biases and couplings on the grid
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <span className="text-3xl font-bold text-purple-300">2.</span>
              <div>
                <h3 className="text-xl font-semibold text-purple-200">
                  Queue Your Actions
                </h3>
                <p className="text-gray-300">
                  Build a queue of moves before committing (you can undo!)
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <span className="text-3xl font-bold text-purple-300">3.</span>
              <div>
                <h3 className="text-xl font-semibold text-purple-200">
                  Mark Ready
                </h3>
                <p className="text-gray-300">
                  When both players are ready, sampling can begin
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <span className="text-3xl font-bold text-purple-300">4.</span>
              <div>
                <h3 className="text-xl font-semibold text-purple-200">
                  Run Sampling
                </h3>
                <p className="text-gray-300">
                  THRML finds the low-energy spin configuration
                </p>
              </div>
            </div>
            <div className="flex items-start gap-4">
              <span className="text-3xl font-bold text-purple-300">5.</span>
              <div>
                <h3 className="text-xl font-semibold text-purple-200">
                  Score & Advance
                </h3>
                <p className="text-gray-300">
                  Player with more cells wins. Biases decay by 50% for next
                  round
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Player Resources */}
        <section className="mb-12 bg-neutral-800/50 rounded-xl p-6 border border-neutral-700">
          <h2 className="text-3xl font-bold mb-4 text-amber-400">
            💰 Player Resources
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-neutral-900/50 rounded-lg p-4 border border-amber-500/30">
              <h3 className="text-xl font-semibold text-amber-300 mb-2">
                🎯 Bias Tokens (2 per round)
              </h3>
              <p className="text-gray-300">
                Modify individual cell preferences to influence which color they
                prefer to be. Click a cell in bias mode to queue a bias change.
              </p>
            </div>
            <div className="bg-neutral-900/50 rounded-lg p-4 border border-emerald-500/30">
              <h3 className="text-xl font-semibold text-emerald-300 mb-2">
                🔗 Edge Tokens (3 per round)
              </h3>
              <p className="text-gray-300">
                Modify coupling between neighboring cells to make them prefer
                alignment or opposition. Click two adjacent cells in edge mode
                to queue a coupling change.
              </p>
            </div>
          </div>
        </section>

        {/* Card System */}
        <section className="mb-12 bg-neutral-800/50 rounded-xl p-6 border border-neutral-700">
          <h2 className="text-3xl font-bold mb-4 text-pink-400">
            🃏 Card System
          </h2>
          <p className="text-lg text-gray-300 mb-4">
            Each player starts with a hand of special cards that provide
            powerful one-time effects:
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-neutral-900/50 rounded-lg p-4 border border-pink-500/30">
              <h4 className="font-semibold text-pink-300">🎯 Bias Cards</h4>
              <ul className="list-disc list-inside text-gray-300 ml-2 mt-2 space-y-1">
                <li>
                  <strong>Strong Bias:</strong> 2x bias effect (costs 1 bias
                  token)
                </li>
                <li>
                  <strong>Ultra Bias:</strong> 3x bias effect (costs 2 bias
                  tokens)
                </li>
              </ul>
            </div>
            <div className="bg-neutral-900/50 rounded-lg p-4 border border-pink-500/30">
              <h4 className="font-semibold text-pink-300">🔗 Coupling Cards</h4>
              <ul className="list-disc list-inside text-gray-300 ml-2 mt-2 space-y-1">
                <li>
                  <strong>Strong Coupling:</strong> 2x coupling effect (costs 1
                  edge token)
                </li>
                <li>
                  <strong>Ultra Coupling:</strong> 3x coupling effect (costs 2
                  edge tokens)
                </li>
              </ul>
            </div>
            <div className="bg-neutral-900/50 rounded-lg p-4 border border-pink-500/30">
              <h4 className="font-semibold text-pink-300">🌟 Special Cards</h4>
              <ul className="list-disc list-inside text-gray-300 ml-2 mt-2 space-y-1">
                <li>
                  <strong>Freeze Cell:</strong> Lock a cell's current state
                </li>
                <li>
                  <strong>Boost Beta:</strong> Increase determinism temporarily
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Controls & Tips */}
        <section className="mb-12 bg-neutral-800/50 rounded-xl p-6 border border-neutral-700">
          <h2 className="text-3xl font-bold mb-4 text-cyan-400">
            🎮 Controls & Tips
          </h2>
          <div className="space-y-4 text-gray-300">
            <div>
              <h3 className="text-xl font-semibold text-cyan-300 mb-2">
                Basic Controls
              </h3>
              <ul className="list-disc list-inside ml-4 space-y-1">
                <li>
                  <strong>Bias Mode:</strong> Click a cell to queue a bias
                  change toward your color
                </li>
                <li>
                  <strong>Edge Mode:</strong> Click two adjacent cells to
                  strengthen their coupling
                </li>
                <li>
                  <strong>Preview:</strong> See predicted outcomes before
                  committing
                </li>
                <li>
                  <strong>Undo:</strong> Remove the last action from your queue
                </li>
                <li>
                  <strong>Commit:</strong> Apply all queued actions and spend
                  tokens
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-cyan-300 mb-2">
                Strategic Tips
              </h3>
              <ul className="list-disc list-inside ml-4 space-y-1">
                <li>
                  Biases persist across rounds (with 50% decay), so think
                  long-term!
                </li>
                <li>Strong couplings can create defensive clusters</li>
                <li>Use the preview feature to test different strategies</li>
                <li>Save special cards for critical moments</li>
                <li>
                  Watch your opponent's budget to predict their capabilities
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Technology */}
        <section className="mb-12 bg-neutral-800/50 rounded-xl p-6 border border-neutral-700">
          <h2 className="text-3xl font-bold mb-4 text-emerald-400">
            🔬 Technology
          </h2>
          <p className="text-gray-300 text-lg">
            This game uses <strong className="text-emerald-300">THRML</strong>{" "}
            (a thermal computing library) with{" "}
            <strong className="text-emerald-300">JAX</strong> for efficient
            Gibbs sampling. The checkerboard blocking strategy enables parallel
            updates for fast computation of the Ising model dynamics.
          </p>
        </section>

        {/* Call to Action */}
        <div className="text-center">
          <Link
            to="/"
            className="inline-block px-8 py-4 bg-gradient-to-r from-emerald-500 to-blue-500 hover:from-emerald-600 hover:to-blue-600 rounded-lg font-bold text-xl transition-all transform hover:scale-105 shadow-lg"
          >
            Start Playing! ⚡
          </Link>
        </div>
      </div>
    </div>
  );
}
