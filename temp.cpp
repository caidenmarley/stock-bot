#include <utility>  // for std::pair

std::pair<Eigen::VectorXd, Eigen::VectorXd>
LSTMCell::backwardPass(const Eigen::VectorXd& delta_h,
                       const Eigen::VectorXd& delta_c) {
    // 1) Grab the last time-step’s cache
    const StepData& sd = this->stepData.back();

    // 2) ----- back-prop through h_t = o ⊙ tanh(c) -----
    // tanh(c_t)
    Eigen::VectorXd tanh_c = sd.c.array().tanh().matrix();

    // δo = δh * tanh(c)
    Eigen::VectorXd d_o = delta_h.array() * tanh_c.array();

    // δc via h: δh * o * (1 - tanh(c)^2)
    Eigen::VectorXd d_c_from_h =
        (delta_h.array() * sd.o.array() 
         * (1.0 - tanh_c.array().square()))
         .matrix();

    // total δc at this time-step
    Eigen::VectorXd delta_c_total = delta_c + d_c_from_h;

    // 3) ----- back-prop through c_t = f * c_{t-1} + i * c_tilde -----
    // δf = δc_total * c_{t-1}
    Eigen::VectorXd d_f = delta_c_total.array() * sd.prevCellState.array();
    // δi = δc_total * c_tilde
    Eigen::VectorXd d_i = delta_c_total.array() * sd.c_tilde.array();
    // δc_tilde = δc_total * i
    Eigen::VectorXd d_c_tilde =
        delta_c_total.array() * sd.i.array();
    // δc_{t-1} for next iteration
    Eigen::VectorXd delta_c_prev =
        (delta_c_total.array() * sd.f.array()).matrix();

    // 4) ----- back-prop through non-linearities -----
    // σ′(z) = σ(z)*(1−σ(z)), tanh′(z) = 1−tanh^2(z)
    Eigen::VectorXd df_pre =
        (d_f.array() * sd.f.array() * (1.0 - sd.f.array())).matrix();
    Eigen::VectorXd di_pre =
        (d_i.array() * sd.i.array() * (1.0 - sd.i.array())).matrix();
    Eigen::VectorXd do_pre =
        (d_o.array() * sd.o.array() * (1.0 - sd.o.array())).matrix();
    Eigen::VectorXd dc_tilde_pre =
        (d_c_tilde.array() * (1.0 - sd.c_tilde.array().square())).matrix();

    // 5) ----- accumulate parameter gradients -----
    // forget gate
    this->dWf.noalias() += df_pre * sd.input.transpose();
    this->dUf.noalias() += df_pre * sd.prevHiddenState.transpose();
    this->dbf.noalias() += df_pre;

    // input gate
    this->dWi.noalias() += di_pre * sd.input.transpose();
    this->dUi.noalias() += di_pre * sd.prevHiddenState.transpose();
    this->dbi.noalias() += di_pre;

    // candidate
    this->dWc.noalias() += dc_tilde_pre * sd.input.transpose();
    this->dUc.noalias() += dc_tilde_pre * sd.prevHiddenState.transpose();
    this->dbc.noalias() += dc_tilde_pre;

    // output gate
    this->dWo.noalias() += do_pre * sd.input.transpose();
    this->dUo.noalias() += do_pre * sd.prevHiddenState.transpose();
    this->dbo.noalias() += do_pre;

    // 6) ----- build δh_{t-1} from the U_* paths -----
    Eigen::VectorXd delta_h_prev =
        this->Uf.transpose() * df_pre
      + this->Ui.transpose() * di_pre
      + this->Uc.transpose() * dc_tilde_pre
      + this->Uo.transpose() * do_pre;

    // 7) pop this step’s cache and return the two previous deltas
    this->stepData.pop_back();
    return {delta_h_prev, delta_c_prev};
}
