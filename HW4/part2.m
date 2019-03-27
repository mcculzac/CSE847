


function [] = part2()

    function [w, c] = logistic_l1_train(data, labels, par)
    % OUTPUT w is equivalent to the first d dimension of weights in logistic train
    % c is the bias term, equivalent to the last dimension in weights in logistic train.
    % Specify the options (use without modification).
        opts.rFlag = 1; % range of par within [0, 1].
        opts.tol = 1e-6; % optimization precision
        opts.tFlag = 4; % termination options.
        opts.maxIter = 5000; % maximum iterations.

        [w, c] = LogisticR(data, labels, par, opts);

    end

    function [predictions] = test_model(w, xtest, ytest)
        right = 0;
        % disp('sizes: ');
        % disp(xtest(1:10));
        % disp(size(w));
        % disp(size(xtest(1)));
        predictions = [];
        for k=1: length(xtest)
            pred_orig = xtest(k, :) * w;
            % disp(pred_orig);
            if pred_orig > 0
                pred = 1;
            elseif pred_orig < 0
                pred = -1;
            else
                % disp('pred_orig is 0?');
                % disp(pred_orig);
                pred = 0;
            end
            predictions = [predictions pred];
            if pred == ytest(k)
                right = right + 1;
            end
        end
        acc = right / length(xtest);
        disp('accuracy ');
        disp(acc);
    end

    function [count] = get_non_zero(w)
        count = 0;
        
        for i = 1:length(w)
            if w(i) == 0
                count = count + 1;
            end
        end
    end
    
    data = load('data.txt');
    labels = load('labels.txt');
    labels(labels<1) = -1;
    t_train = labels(1:2000, :);
    t_test = labels(2001:end, :);
    data_train = data(1:2000, :);
    data_test = data(2001:end, :);
    pars = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
    
    
    
    for j = 1:length(pars)
        our_w = logistic_l1_train(data_train, t_train, pars(j));
        disp('Running for par: ');
        disp(pars(j));
        disp('Features selected: ');
        disp(get_non_zero(our_w));
        % disp(t_test(1:10, :));
        preds = test_model(our_w, data_test, t_test);
        [x,y,t,auc] = perfcurve(t_test, preds, 1);
        disp('AUC: ');
        disp(auc);
    end
    

end