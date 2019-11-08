using LinearAlgebra, DataFrames, CSV, Query

cc = CSV.read("UCI_Credit_Card.csv")
cols = names(cc)
println(describe(cc))

## Data Preparation

### SEX
#### 1 = Male, 2 = Female
#### New: 0 = Male, 1 = Female
unique(cc[:SEX])
cc[:SEX] = map(s -> ifelse(s == 1, 0, 1), cc.SEX)

### EDUCATION
#### 1 = graduate school; 2 = university; 3 = high school; 4 = others
#### New: 1 = graduate school; 2 = university; 3 = high school; 4 = others, 5 = unknown
sort(unique(cc.EDUCATION))
cc[:EDUCATION] = map(s -> ifelse(s in (1, 2, 3, 4), s, 5), cc.EDUCATION)
sort(unique(cc.EDUCATION))

### MARRIAGE
####  1 = married; 2 = single; 3 = others
#### New: 1 => married 1; 2 => single 2; 3 => others 3; rest => others 3
sort(unique(cc.MARRIAGE))
cc[:MARRIAGE] = map(s -> ifelse(s in (1, 2, 3), s, 4), cc.MARRIAGE)
sort(unique(cc.MARRIAGE))


### PAY_0, 2, 3, 4, 5, 6
#### -1, 1, 2, 3, 4, 5, 6, 7, 8, 9
#### New -2, -1, 0 => 0; 1, 2, 3, 4, 5, 6, 7, 8, 9 => 1, 2, 3, 4, 5, 6, 7, 8, 9
for i in [0, 2, 3, 4, 5, 6]
    #println(Symbol("PAY_", i))
    cc[Symbol("PAY_", i)] = map(s -> ifelse(s in (-2, -1), 0, s),
                                cc[Symbol("PAY_", i)])
end

println(describe(cc))

# One Hot Encoding

function OneHot(x::Vector{Int64}, lab::Symbol)
    p = sort(unique(x))
    colname = string(lab)
    OHdf = DataFrame(keys = p)

    for i in 1:length(p)
        val = p[i]
        v = map(s -> ifelse(s == val, 1, 0), OHdf.keys)
        insertcols!(OHdf, i+1, Symbol(colname, val) => v)
    end
    OHdf
end


# Education variable
edu = Vector(cc[:EDUCATION])
lab = :EDUCATION
M1 = OneHot(edu, lab)
M1

# Marriage variable
marriage = Vector(cc[:MARRIAGE])
lab = :MARRIAGE
M2 = OneHot(marriage, lab)
M2

cc = join(cc, M1, on = :EDUCATION => :keys)

println(describe(cc))

cc = join(cc, M2, on = :MARRIAGE => :keys)

println(describe(cc))

# Data Cleaning Done,
# DROP columns EDUCATION, MARRIAGE,
# Set reference group by dropping: EDUCATION1, MARRIAGE1
# Thus, reference group becomes: Male, Grad School, Married
# REORDER

cols = names(cc)
keepcols = [:ID,
            :LIMIT_BAL,
            :SEX,
            :AGE,
            :EDUCATION2,
            :EDUCATION3,
            :EDUCATION4,
            :EDUCATION5,
            :MARRIAGE2,
            :MARRIAGE3,
            :MARRIAGE4,
            :PAY_0,
            :PAY_2,
            :PAY_3,
            :PAY_4,
            :PAY_5,
            :PAY_6,
            :BILL_AMT1,
            :BILL_AMT2,
            :BILL_AMT3,
            :BILL_AMT4,
            :BILL_AMT5,
            :BILL_AMT6,
            :PAY_AMT1,
            :PAY_AMT2,
            :PAY_AMT3,
            :PAY_AMT4,
            :PAY_AMT5,
            :PAY_AMT6,
            Symbol("default.payment.next.month")]

cc = cc[keepcols]

CSV.write("CCDataClean.csv", cc)
