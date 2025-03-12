function z = smooth_square_sum(X,listUnique)
%% Performs the f^T*L*f of clique/star expansion
% i.e., the summation of the squared differences of signals

z = [];  % Distance vector
for j = 1:length(listUnique)  % For each list of combinations of hyperedges per cardinality
    signals = X(listUnique{j},:);  % Each row will be all the signal observations of one of the nodes in hyperedge combination list{i}(j)

    % Pairwise distance calculations
    iter = 2;  % Handles the combination of nodes
    summation = 0;  % Tracks sum
    for a = 1:size(signals,1)-1  % From node 1 to N-1
        for b = iter:size(signals,1)  % From node 2:N, then 3:N, etc.
            diff = signals(a,:) - signals(b,:);  % Difference between signals
            squared_distances = diff * diff';  % Sum of differences squared
            summation = summation + (squared_distances);  % Summation of differences squared for all observations of this pair of nodes
        end
        iter = iter + 1;
    end

    z(end+1,1) = summation;  % Store the distances

end


end