function z = smooth_abs_sum(X,listUnique)
%% Uses the sum of the absolute pairwise difference between all signals in the hyperedge
%  sum_i,j |x_i - x_j|

z = [];  % Distance vector

for j = 1:length(listUnique)  % For each list of combinations of hyperedges per cardinality
    signals = X(listUnique{j},:);  % Each row will be all the signal observations of one of the nodes in hyperedge combination list{i}(j)

    % Pairwise distance calculations
    iter = 2;  % Handles the combination of nodes
    summation = 0;  % Tracks sum
    for a = 1:size(signals,1)-1  % From node 1 to N-1
        for b = iter:size(signals,1)  % From node 2:N, then 3:N, etc.
            diff = sum(abs(signals(a,:) - signals(b,:)));  % Absolute difference
            summation = summation + diff;  % Summations of all absolute differences for hyperedge
        end
        iter = iter + 1;
    end

    z(end+1,1) = summation;  % Store the distances

end


end