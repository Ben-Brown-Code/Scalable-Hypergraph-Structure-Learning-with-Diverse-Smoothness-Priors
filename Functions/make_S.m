function S = make_S(listUnique, N)

Da = length(listUnique);

S = [];
for n = 1:N
    temp = zeros(1,Da);
    for k = 1:Da
        if ismember(n,listUnique{k})
            temp(k) = 1;  % Each row (node) will contain ones in positions that correspond to the hyperedges (Da) it could belong to
        end
    end
    S = [S;temp];
end

end