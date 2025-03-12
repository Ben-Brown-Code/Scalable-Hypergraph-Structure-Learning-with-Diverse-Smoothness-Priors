function listUnique = generate_knn_hyperedges(X, num_neighbors)
    % Generate hyperedges using KNN with varying hyperedge sizes.
    %
    % Parameters:
    %   X: N x P matrix, where N is the number of nodes, P is the length of time series.
    %   Num_neighbors: A vector specifying the number of neighbors for hyperedges of different sizes.
    %                 The first element corresponds to hyperedges of size 2, the second to size 3, etc.
    %
    % Returns:
    %   listUnique: A cell array where each element is a list of node indices forming a hyperedge.
    
    N = size(X, 1); % Number of nodes
    listUnique = {};
    
    for edge_size = 2:length(num_neighbors)+1
        k = num_neighbors(edge_size-1); % Number of neighbors for this hyperedge size
        
        % Use knnsearch to find nearest neighbors
        idx_raw = knnsearch(X, X, 'K', k+1);
        idx_raw = idx_raw(:, 2:end); % Exclude self
        
        for i = 1:N
            neighbors = idx_raw(i, 1:k); % Ensure k neighbors are used
            hyperedges = nchoosek(neighbors, edge_size-1); % Form hyperedges of required size
            
            for j = 1:size(hyperedges, 1)
                hyperedge = sort([i, hyperedges(j, :)]); % Include the source node
                hyperedge_cell = num2cell(hyperedge,2); % Convert to cell format
                
                if ~any(cellfun(@(x) isequal(x, hyperedge_cell{1}), listUnique))
                    listUnique(end+1) = hyperedge_cell;
                end
            end
        end
    end
    listUnique = listUnique';
end
