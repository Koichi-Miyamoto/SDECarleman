using LinearAlgebra

function MakeKronSum(A, j)

  nRowA = size(A, 1)
  nColA = size(A, 2)
  In = Matrix{eltype(A)}(I, nRowA, nRowA)

  result = zeros(nRowA^j, nRowA^(j-1) * nColA)

  for k in 1:j
    term = k == 1 ? A : In

    for l in 2:j
      factor = (l == k) ? A : In
      term = kron(term, factor)
    end

    result += term
  end

  return result

end

function MakeA_j_jminus2(C, j)
  d = size(C, 1)

  @assert size(C, 2) == d
  @assert j >= 2

  A = zeros(d^j, d^(j-2))

  LI_out = LinearIndices(ntuple(_ -> d, j))
  LI_in  = LinearIndices(ntuple(_ -> d, j-2))

  for k in 1:j-1
    for l in k+1:j
      for Iprime in CartesianIndices(ntuple(_ -> d, j-2))
        ip = Tuple(Iprime)
        for a in 1:d
          for b in 1:d

            i = Vector{Int}(undef, j)
            q = 1

            for r in 1:j
              if r == k
                i[r] = a
              elseif r == l
                i[r] = b
              else
                i[r] = ip[q]
                q += 1
              end
            end

            row = LI_out[CartesianIndex(Tuple(i))]
            col = LI_in[Iprime]

            A[row, col] += C[a, b]
          end
        end
      end
    end
  end

  return A
end

function MakeCalC(A, B, F, J)

  n = size(A, 1)
  @assert size(A, 2) == n
  nPows = n .^ (1:J)
  blockEndIds = cumsum([1; nPows[1:end-1]])
  nTot = sum(nPows)
  result = zeros(nTot, nTot)
  
  # diagonal
  for j in 1:J
    iRow = iCol = blockEndIds[j]
    nRow = nCol = nPows[j]
    result[iRow:iRow+nRow-1, iCol:iCol+nCol-1] = MakeKronSum(A, j)
  end

  # subdiagonal
  BBT = B * B'
  for j in 3:J
    iRow = blockEndIds[j]
    iCol = blockEndIds[j-2]
    nRow = nPows[j]
    nCol = nPows[j-2]
    result[iRow:iRow+nRow-1, iCol:iCol+nCol-1] = MakeA_j_jminus2(BBT, j)
  end

  # superdiagonal
  temp = size(F, 2)
  s = 0
  while temp % n == 0
    temp = div(temp, n)
    s += 1
  end
  @assert temp == 1
  for j in 1:J-s+1
    iRow = blockEndIds[j]
    iCol = blockEndIds[j+s-1]
    nRow = nPows[j]
    nCol = nPows[j+s-1]
    result[iRow:iRow+nRow-1, iCol:iCol+nCol-1] = MakeKronSum(F, j)
  end

  return result

end

function MakeVecc(B, J)

  n = size(B, 1)
  nTot = n * div(n^J - 1, n - 1)
  result = zeros(nTot)
  result[n+1:n+n^2] = vec(B * B')
  return result

end

function KronAppend(v, J)

    w = [one(eltype(v))]
    result = eltype(v)[]

    for _ in 1:J
        w = kron(w, v)
        append!(result, w)
    end

    return result

end