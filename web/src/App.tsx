import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Game } from "./pages/Game";
import { HowToPlay } from "./pages/HowToPlay";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Game />} />
        <Route path="/how-to-play" element={<HowToPlay />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
