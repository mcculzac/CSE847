
function CSE847HW42
    A = [];
    L=[];
    load('USPS.mat');
    function [error] = pca_of(p, A)
        
        [cof,s,l,t,e,mu] = pca(A);
        reconstruct = s(:, 1:p)*cof(:,1:p)'+repmat(mu, 3000, 1);
        for i=1:2
            a = reshape(A(i,:), 16, 16);
            subplot(2, 2, i);
            imshow(a');
        end
        for j=1:2
            subplot(2, 2, j+2);
            b = reshape(reconstruct(j,:), 16, 16);
            imshow(b');
        end
        error = sum(sum((A - reconstruct).^2));
    end
    
    ps = [10, 50, 100 200];
    for k=1:4
        disp('ps:');
        ps(k);
        disp('Error:');
        disp(pca_of(ps(k), A));
    end
end