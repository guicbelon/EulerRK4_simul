% Constantes e parâmetros iniciais
global g m1 m2 l1 l2 time_step simulation_time theta_1_initial theta_2_initial theta_1_dot_initial theta_2_dot_initial theta_1_dot_dot_initial theta_2_dot_dot_initial;

m1 = 0.5;
m2 = 1.0;
l1 = 1.0;
l2 = 1.5;
time_step = 0.001;
g = 9.8;

simulation_time = 60;

n_steps = simulation_time / time_step;

theta_1_initial = 0;
theta_2_initial = 0;
theta_1_dot_initial = 0.05;
theta_2_dot_initial = 0.1;
theta_1_dot_dot_initial = 0;
theta_2_dot_dot_initial = 0;

% Funções para cálculo das acelerações angulares
function theta_1_dot_dot_value = theta_1_dot_dot(theta_1, theta_1_dot, theta_2, theta_2_dot)
    global g m1 m2 l1 l2
    num = (-sin(theta_1 - theta_2) * (m2 * l2 * (theta_2_dot^2) + m2 * l1 * (theta_1_dot^2) * cos(theta_1 - theta_2)) - g * ((m1 + m2) * sin(theta_1) - m2 * (sin(theta_2) * cos(theta_1 - theta_2))));
    den = l1 * (m1 + m2 * ((sin(theta_1 - theta_2))^2));
    theta_1_dot_dot_value = num / den;
end

function theta_2_dot_dot_value = theta_2_dot_dot(theta_1, theta_1_dot, theta_2, theta_2_dot)
    global g m1 m2 l1 l2
    num = (sin(theta_1 - theta_2) * ((m1 + m2) * l1 * (theta_1_dot^2) + m2 * l2 * (theta_2_dot^2) * cos(theta_1 - theta_2)) + g * ((m1 + m2) * (sin(theta_1) * cos(theta_1 - theta_2) - sin(theta_2))));
    den = l2 * (m1 + m2 * ((sin(theta_1 - theta_2))^2));
    theta_2_dot_dot_value = num / den;
end

% Função para aplicar as equações de movimento
function next_state = apply_f(state)
    global time_step
    % Extrair os valores do estado atual
    theta_1 = state(1);
    theta_1_dot = state(2);
    theta_2 = state(4);
    theta_2_dot = state(5);

    % Calcular o próximo valor de theta_1
    next_theta_1 = theta_1 + time_step * theta_1_dot;
    next_theta_1_dot_dot = theta_1_dot_dot(theta_1, theta_1_dot, theta_2, theta_2_dot);

    next_theta_1_dot = theta_1_dot + time_step * next_theta_1_dot_dot;

    next_theta_2 = theta_2 + time_step * theta_2_dot;
    next_theta_2_dot_dot = theta_2_dot_dot(theta_1, theta_1_dot, theta_2, theta_2_dot);
    next_theta_2_dot = theta_2_dot + time_step * next_theta_2_dot_dot;
    next_state = [next_theta_1, next_theta_1_dot, next_theta_1_dot_dot, next_theta_2, next_theta_2_dot, next_theta_2_dot_dot];

    next_state(isnan(next_state)) = 0;
    next_state(isinf(next_state)) = 100 0;
    %next_state = round(next_state, 4);

end

% Função principal para simulação
function states = simul()
    global theta_1_initial theta_1_dot_initial theta_2_initial theta_2_dot_initial theta_1_dot_dot_initial theta_2_dot_dot_initial time_step simulation_time
    % Pré-alocar matriz de estados (mais uma linha para incluir o estado inicial)
    n_steps = simulation_time / time_step;
    states = zeros(n_steps + 1, 6);

    state = [theta_1_initial, theta_1_dot_initial, theta_1_dot_dot_initial, theta_2_initial, theta_2_dot_initial, theta_2_dot_dot_initial];

    % Armazenar o estado inicial na primeira linha da matriz states
    states(1, :) = state;
    for i = 1:n_steps
        disp(['itr ',num2str(i)]);
        disp(state);
        k1 = apply_f(state);  % Atualiza o estado usando a função apply_f

        state_k2 = state +0.5*time_step*k1;
        k2 = apply_f(state_k2);

        state_k3 = state + 0.5*time_step*k2;
        k3 = apply_f(state_k3);

        state_k4 = state + time_step*k3;
        k4 = apply_f(state_k4);

        state = state + (1/6)*time_step*(k1 + 2*k2 + 2*k3 + k4);

        states(i, :) = state;    % Armazena o estado atual na matriz de estados
    end
end

simulated_states = simul();

