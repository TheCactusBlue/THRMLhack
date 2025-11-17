import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Game } from "./pages/Game";
import { HowToPlay } from "./pages/HowToPlay";
import { ParticlesBackground } from "./components/ParticlesBackground";

function App() {
  return (
    <BrowserRouter>
      <ParticlesBackground />
      <Routes>
        <Route path="/" element={<Game />} />
        <Route path="/how-to-play" element={<HowToPlay />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
