function z = smooth_abs_max(X,listUnique)
%% Uses the maximum of the absolute pairwise difference between all signals in the hyperedge
%  max_i,j |x_i - x_j|

z = [];  % Distance vector
for j = 1:length(listUnique)  % For each list of combinations of hyperedges per cardinality
    signals = X(listUnique{j},:);  % Each row will be all the signal observations of one of the nodes in hyperedge combination list{i}(j)

    % Pairwise distance calculations
    iter = 2;  % Handles the combination of nodes
    temp = [];  % Stores differences to be compared
    for a = 1:size(signals,1)-1  % From node 1 to N-1
        for b = iter:size(signals,1)  % From node 2:N, then 3:N, etc.
            temp(end+1) = sum(abs(signals(a,:) - signals(b,:)));  % Absolute difference, stored for later comparison. The sum adds all elements of the difference vector
        end
        iter = iter + 1;
    end

    z(end+1,1) = max(temp);  % Store the max absolute distance from each difference combination

end

