function z = smooth_square_max(X,listUnique)
%% Performs the f^T*L*f of clique/star expansion with max
% i.e., the summation of the squared differences of signals using the max

z = [];  % Distance vector
for j = 1:length(listUnique)  % For each list of combinations of hyperedges per cardinality
    signals = X(listUnique{j},:);  % Each row will be all the signal observations of one of the nodes in hyperedge combination list{i}(j)

    % Pairwise distance calculations
    iter = 2;  % Handles the combination of nodes
    temp = [];  % Stores differences
    for a = 1:size(signals,1)-1  % From node 1 to N-1
        for b = iter:size(signals,1)  % From node 2:N, then 3:N, etc.
            diff = signals(a,:) - signals(b,:);  % Difference between signals
            temp(end+1) = diff * diff';  % Sum of differences squared, stored
        end
        iter = iter + 1;
    end

    z(end+1,1) = max(temp);  % Store the distances

end






end