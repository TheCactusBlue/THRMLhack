export interface PlayerBudget {
  edge_tokens: number
  bias_tokens: number
  edge_tokens_used: number
  bias_tokens_used: number
}

export interface GameState {
  grid_size: number
  biases: number[]
  couplings: number[]
  beta: number
  last_board?: number[][]
  spin_confidence?: number[][]
  energy?: number
  magnetization?: number
  player_a_territory?: number
  player_b_territory?: number
  current_round: number
  player_a_budget?: PlayerBudget
  player_b_budget?: PlayerBudget
  player_a_ready: boolean
  player_b_ready: boolean
  player_a_wins: number
  player_b_wins: number
  max_rounds: number
  game_winner?: string | null
}

export type PlayerType = 'A' | 'B'
