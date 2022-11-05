import HomePage from "./pages/home";
import LoginPage from "./pages/Login";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import PrivateRoute from "./utils/PrivateRoute";
import { AuthProvider } from "./context/AuthContext";
import Navigation from "./components/Navigation";

function App() {
  return (
    <div className="App">
      
        <Router>
        <AuthProvider>
          <Navigation />
          <Routes>
            <Route element={<HomePage />} path="/" exact />
            <Route element={<LoginPage />} path="/login" exact />
          </Routes>
          </AuthProvider>
        </Router>
      
    </div>
  );
}

export default App;
