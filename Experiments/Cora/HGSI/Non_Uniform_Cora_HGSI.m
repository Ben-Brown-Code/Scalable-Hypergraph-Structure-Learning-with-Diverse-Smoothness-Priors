clear
global_timer = tic;

%% Load Ground Truth
load('incidence_matrix_connected_4_numEdges=48_N=61_M=3.mat')  % Non-Uniform Cora

N = size(H,1); % Number of nodes
M = 3;  % Max cardinality

%% Generate Signal
observations = 250;  % Number of signal observations
L = incidence_laplacian(H);  % Creates Laplacian from incidence matrix
[X_v,~] = Bipartite_Signal(L,observations,N);  % Returns matrix of signal observations

%% HGSI Learning Algorithm Call
K = [2,3];
num_hyperedges = size(H,2);
[learned_edges,learned_weights,w,listUnique,z] = HGSI_algorithm(X_v,K,num_hyperedges);

%% Metrics
C = cells_from_incidence(H);

[A, precision, recall, F1] = metrics(learned_edges,C);
overall_stats = [A;precision;recall;F1];
%%
total_time = toc(global_timer);
fprintf('Total Time Elapsed: %.2f\n', total_time);