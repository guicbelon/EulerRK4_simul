
g = 9.8;
m1 = 0.5;
m2 = 1.0;
l1 = 1.0;
l2 = 1.5;
time_step = 0.01;
simulation_time = 60;

n_steps = simulation_time/time_step;


theta_1_initial=0;
theta_2_initial= 0;
theta_1_dot_initial = 0.05;
theta_2_dot_initial = 0.1;
theta_1_dot_dot_initial = 0; 
theta_2_dot_dot_initial = 0; 


function theta_1_dot_dot = theta_1_dot_dot(theta_1, theta_1_dot, theta_2, theta_2_dot)
    num = (-sin(theta_1-theta_2)*(m2*l2*(theta_2_dot^2) + m2*l1*(theta_1_dot^2)*cos(theta_1-theta_2)) - g*((m1 + m2)*sin(theta_1) - m2*(sin(theta_2)*cos(theta_1-theta_2))));
    den = l1*(m1 + m2*((sin(theta_1-theta_2))^2));
    theta_1_dot_dot = num / den;
end

function theta_2_dot_dot = theta_2_dot_dot(theta_1, theta_1_dot, theta_2, theta_2_dot)
    num = (sin(theta_1-theta_2)*((m1+m2)*l1*(theta_1_dot^2) + m2*l2*(theta_2_dot^2)*cos(theta_1-theta_2)) + g*((m1 + m2)*(sin(theta_1)*cos(theta_1-theta_2)-sin(theta_2))));
    den = l2*(m1 + m2*((sin(theta_1-theta_2))^2));
    theta_2_dot_dot = num / den;
end


function next_state = apply_f(state)
    theta_1 = state(1);
    theta_1_dot = state(2);
    theta_2 = state(4);
    theta_2_dot = state(5); 
    
    next_theta_1 = theta_1 + time_step*theta_1_dot;
    next_theta_1_dot_dot = theta_1_dot_dot(theta_1, theta_1_dot, theta_2, theta_2_dot);  
    next_theta_1_dot = theta_1_dot + time_step*next_theta_1_dot_dot;
    
    next_theta_2 = theta_2 + time_step*theta_2_dot;
    next_theta_2_dot_dot = theta_2_dot_dot(theta_1, theta_1_dot, theta_2, theta_2_dot);  
    next_theta_2_dot = theta_2_dot + time_step*next_theta_2_dot_dot;
    
    next_state = [next_theta_1, next_theta_1_dot, next_theta_1_dot_dot, next_theta_2, next_theta_2_dot, next_theta_2_dot_dot];
    
    % Note: Uncomment below lines for handling NaNs, Infs, or rounding if needed
    % next_state(isnan(next_state)) = 0;
    % next_state(isinf(next_state)) = 0;
    % next_state = round(next_state, 4);
end


function states = simul()

    state = [theta_1_initial, theta_1_dot_initial, theta_1_dot_dot_initial, theta_2_initial, theta_2_dot, theta_2_dot_initial];
    states = [array2];
    for i = 1:n_steps
        k1 = apply_f(state)
        disp(k1)
        
    end