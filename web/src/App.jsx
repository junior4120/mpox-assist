import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Analyse from "./pages/Analyse";
import Historique from "./pages/Historique";
import Modele from "./pages/Modele";
import MobileApp from "./pages/MobileApp";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/mobile" element={<MobileApp />} />
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyse" element={<Analyse />} />
          <Route path="/historique" element={<Historique />} />
          <Route path="/modele" element={<Modele />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}