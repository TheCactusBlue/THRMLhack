// Card system types
export type CardType =
  | 'infiltrate'
  | 'disruption'
  | 'fortress'
  | 'anchor'
  | 'heat_wave'
  | 'freeze'

export interface Card {
  type: CardType
  name: string
  description: string
  bias_cost: number
  edge_cost: number
}

export interface PlayerBudget {
  edge_tokens: number
  bias_tokens: number
  edge_tokens_used: number
  bias_tokens_used: number
  hand: string[]  // Card types in hand
  played_cards: string[]  // Card types played this round
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
  entrenchment?: number[][]  // PHASE 2: Rounds of consecutive control per cell
}

// PHASE 2: Action for the undo queue
export interface Action {
  id: string
  type: 'bias' | 'coupling'
  description: string
  params: {
    row?: number
    col?: number
    cell1?: [number, number]
    cell2?: [number, number]
    direction: number
    player: string
  }
}

export type PlayerType = 'A' | 'B'
