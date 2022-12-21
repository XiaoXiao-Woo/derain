function [Fx Fy] = get_f(h, w)
  all_ids = reshape([1:h*w], [h w]);

  self_ids=all_ids;
  negh_ids=circshift(all_ids, [0 -1]);
  ind=ones(h,w);
  S_plus=sparse(self_ids(:), self_ids(:), ind);
  S_minus=sparse(self_ids(:), negh_ids(:), ind);
  Fx = S_plus-S_minus;
  
  negh_ids=circshift(all_ids, [-1 0]);
  ind=ones(h,w);
  S_plus=sparse(self_ids(:), self_ids(:), ind);
  S_minus=sparse(self_ids(:), negh_ids(:), ind);
  Fy = S_plus-S_minus;



